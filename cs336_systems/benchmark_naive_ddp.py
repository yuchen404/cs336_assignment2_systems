import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"   # Adjust as needed
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from copy import deepcopy

from typing import Iterable, Set, Tuple, List, Dict, Any
from cs336_systems.ddp_utils import (
    get_ddp_individual_parameters,
    ddp_individual_parameters_on_after_backward,
)
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM

# ======= helper =======
def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

# --- Distributed Setup and Teardown ---
def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def teardown_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

# ===================
# DDP Training Function
# ===================
def train_ddp_naive(model: nn.Module, optimizer: torch.optim.Optimizer, input_x, target_y, device: torch.device):
    model.train()
    cuda_sync_if_needed(device)

    t0 = timer()

    # --- forward ---
    logits = model(input_x)
    loss = cross_entropy(logits, target_y)

    # --- backward ---
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # --- DDP gradient synchronization ---
    cuda_sync_if_needed(device)
    comm_s = timer()

    ddp_individual_parameters_on_after_backward(model)

    cuda_sync_if_needed(device)
    comm_e = timer()

    # --- optimizer step ---
    optimizer.step()

    cuda_sync_if_needed(device)
    t1 = timer()

    total_time = t1 - t0
    comm_time = comm_e - comm_s

    return total_time, comm_time

# ===================
# DDP Worker
# ===================
def worker(
    rank: int, world_size: int, backend: str, master_addr: str, master_port: str, 
    vocab_size: int, 
    context_length: int, 
    gloabl_batch_size: int,
    dtype_str: str,
    warmup_steps: int,
    measure_steps: int,
    ) -> None:
    # --- setup process group ---
    setup_process_group(rank, world_size, backend, master_addr, master_port)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    device_str = f"cuda:{rank}"

    if dtype_str == "bf16":
        dtype = torch.bfloat16
    elif dtype_str == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    assert gloabl_batch_size % world_size == 0, "Global batch size must be divisible by world size"
    local_batch_size = gloabl_batch_size // world_size

    # --- model init ---
    torch.manual_seed(rank)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    ).to(device=device, dtype=dtype)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # --- broadcast model parameters from rank 0 ---
    dist.barrier()
    ddp_model = get_ddp_individual_parameters(model, src=0)
    dist.barrier()

    g = torch.Generator().manual_seed(42)
    data_random = torch.randint(0, vocab_size, (1 << 12, ), generator=g).numpy()

    # --- warmup ---
    for step in range(warmup_steps):
        # --- random input ---
        input_x, target_y = get_batch(
            dataset=data_random,
            batch_size=local_batch_size,
            context_length=context_length,
            device=device_str,
        )
        _ = train_ddp_naive(ddp_model, optimizer, input_x, target_y, device)
    dist.barrier()

    # --- measurement ---
    per_step_times = []
    for _ in range(measure_steps):
        input_x, target_y = get_batch(
            dataset=data_random,
            batch_size=local_batch_size,
            context_length=context_length,
            device=device_str,
        )
        step_time, comm_time = train_ddp_naive(ddp_model, optimizer, input_x, target_y, device)
        per_step_times.append({"step_time": step_time, "comm_time": comm_time, "backend": backend})

    # --- gather times results to rank 0---
    gathered_times: List[List[Dict[str, float]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_times, per_step_times)

    if rank == 0:
        # per rank results
        all_ranks_times: List[Dict[str, float]] = []
        all_ranks_times = [item for rank_rec in gathered_times for item in rank_rec]

        # aggregate results
        total_times = [rec["step_time"] for rec in all_ranks_times]
        comm_times = [rec["comm_time"] for rec in all_ranks_times]
        avg_total_time = sum(total_times) / len(total_times)
        avg_comm_time = sum(comm_times) / len(comm_times)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_rank_md = f"results/benchmark_naive_ddp/naive_ddp_benchmark_rank_{backend}_ws{world_size}_{timestamp}.md"
        write_md(output_rank_md, all_ranks_times)
        print(f"Naive DDP Benchmark (backend={backend}, world_size={world_size}):")
        print(f"  Average Step Time: {avg_total_time:.6f} sec")
        print(f"  Average Communication Time: {avg_comm_time:.6f} sec")
        


def write_md(path_md: str, rows: List[Dict[str, Any]]) -> None:
    # os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    # df.sort_values(by=["size_mb", "world_size", "backend"], inplace=True)
    # df.to_csv(path_csv, index=False)
    # print(f"Saved aggregated results to {path_csv}")
    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w") as f:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))


# ===================
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"   # Adjust as needed

    world_size = 2
    backend = "nccl"
    master_addr = "localhost"
    master_port = "29500"

    vocab_size = 10000
    context_length = 128
    global_batch_size = 8
    dtype_str = "fp16"
    warmup_steps = 5
    measure_steps = 10

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(
            world_size,
            backend,
            master_addr,
            master_port,
            vocab_size, 
            context_length,
            global_batch_size,
            dtype_str,
            warmup_steps,
            measure_steps
        ),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()