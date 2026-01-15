import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Adjust as needed
from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict


from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp_utils import (
    cuda_sync_if_needed, 
    _unique_params_by_storage, 
    _broadcast_params, 
    ddp_flattened_parameters_on_after_backward,
    setup_process_group,
    teardown_process_group,
    )
from cs336_systems.ddp_overlap import DDPBucketedParameters, DDPIndividualParameters
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.optimizer_state_sharding import ShardedOptimizer

# --- Benchmarking Utilities ---
def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def params_bytes(model: nn.Module) -> int:
    params = _unique_params_by_storage(model.parameters())
    total_bytes = sum([tensor_bytes(p.data) for p in params])
    return total_bytes

def grads_bytes(model: nn.Module) -> int:
    params = _unique_params_by_storage(model.parameters())
    total_bytes = sum(
        [tensor_bytes(p.grad) for p in params if p.grad is not None]
    )
    return total_bytes

def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """
    Bytes of tensor states in the optimizer
    """
    local = getattr(optimizer, "_local_optimizer", None)
    if local is not None:
        optimizer = local   

    total_bytes = 0
    for state in optimizer.state.values():  # get the state dict for each parameter
        for v in state.values():    # each state value (e.g., exp_avg, ...) in the state dict
            if isinstance(v, torch.Tensor):
                total_bytes += tensor_bytes(v)
    # for _, st in optimizer.state.items():
    #     if isinstance(st, dict):
    #         for v in st.values():
    #             if torch.is_tensor(v):
    #                 total_bytes += tensor_bytes(v)
    return total_bytes

def cuda_memory_snapshot(device: torch.device):
    """
    Get current allocated CUDA memory in MiB.
    """
    if device.type != "cuda":
        return {"alloc_MiB": 0.0, "reserved_MiB": 0.0, "peak_alloc_MiB": 0.0, "peak_reserved_MiB": 0.0}
    
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "alloc_MiB": alloc_mb,
        "reserved_MiB": reserved_mb,
        "peak_alloc_MiB": peak_alloc_mb,
        "peak_reserved_MiB": peak_reserved_mb,
    }
@dataclass
class MemRecord:
    rank: int
    mode: str
    stage: str
    params_MiB: float
    opt_MiB: float
    grads_MiB: float
    peak_alloc_MiB: float
    peak_reserved_MiB: float
    residual_MiB: float

@dataclass
class TimeRecord:
    rank: int
    mode: str
    step: int
    time_sec: float

def get_peak_record(rank: int, mode: str, stage: str, device: torch.device, model: nn.Module, optimizer: torch.optim.Optimizer) -> MemRecord:
    """
    Get a record of peak memory usage and model/optimizer sizes.
    """
    mem_snapshot = cuda_memory_snapshot(device)
    params_bytes_mb = params_bytes(model) / (1024**2)
    optim_bytes_mb = optimizer_state_bytes(optimizer) / (1024**2)
    grads_bytes_mb = grads_bytes(model) / (1024**2)

    residual_mb = max(0.0, mem_snapshot["peak_alloc_MiB"] - params_bytes_mb - optim_bytes_mb - grads_bytes_mb)
    
    record = MemRecord(
        rank=rank,
        mode=mode,
        stage=stage,
        params_MiB=params_bytes_mb,
        opt_MiB=optim_bytes_mb,
        grads_MiB=grads_bytes_mb,
        peak_alloc_MiB=mem_snapshot["peak_alloc_MiB"],
        peak_reserved_MiB=mem_snapshot["peak_reserved_MiB"],
        residual_MiB=residual_mb,
    )
    return record

# --- Training function ---
def run_once(
    rank: int, world_size: int, backend: str, mode: str, grad_comm_mode: str,
    vocab_size: int, context_length: int, global_batch_size: int, dtype: torch.dtype,
    warmup_steps: int, measure_steps: int,
    ):
    # Set device
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    device_str = f"cuda:{rank}" if backend == "nccl" else "cpu"

    # reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- model init ---
    torch.manual_seed(42)
    base_model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=48,
        d_model=1600, 
        num_heads=25,
        d_ff=6400,
        rope_theta=10_000.0,
    ).to(device=device, dtype=dtype)

    if grad_comm_mode == "bucketed":
        model = DDPBucketedParameters(base_model, bucket_size_mb=100, src_rank=0)
    elif grad_comm_mode == "indiv_overlap":
        model = DDPIndividualParameters(base_model, src_rank=0)
    else:
        _broadcast_params(base_model, src=0)
        model = base_model

    model.train()

    # --- optimizer init ---
    if mode == "sharded":
        optimizer = ShardedOptimizer(model.parameters(), optimizer_cls=AdamW, lr=1e-4)
    elif mode == "full":
        optimizer = AdamW(model.parameters(), lr=1e-4)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # --- synthetic data ---
    g = torch.Generator().manual_seed(42 + rank)
    local_batch_size = global_batch_size // world_size
    data_random = torch.randint(0, vocab_size, (1<<12,), generator=g).numpy()

    # records
    mem_records: List[MemRecord] = []
    time_records: List[TimeRecord] = []

    # 1. report memory after init
    cuda_sync_if_needed(device)
    mem_records.append(get_peak_record(rank, mode, "after_init", device, model, optimizer))

    # --- warm up ---
    for step in range(warmup_steps):
        input_x, target_y = get_batch(dataset=data_random, batch_size=local_batch_size, context_length=context_length, device=device_str)

        # forward
        logits = model(input_x)
        loss = cross_entropy(logits, target_y)
        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # grad sync
        ddp_flattened_parameters_on_after_backward(model)
        # optimizer step
        optimizer.step()

    cuda_sync_if_needed(device)
    dist.barrier()

    # --- measurement ---
    for step in range(measure_steps):
        # reset peak memory stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        input_x, target_y = get_batch(dataset=data_random, batch_size=local_batch_size, context_length=context_length, device=device_str)

        # forward
        logits = model(input_x)
        loss = cross_entropy(logits, target_y)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # grad sync
        ddp_flattened_parameters_on_after_backward(model)

        # 2. record memory before optimizer step
        cuda_sync_if_needed(device)
        mem_records.append(get_peak_record(rank, mode, "pre_opt", device, model, optimizer))

        # optimizer step
        t0 = timer()
        optimizer.step()

        cuda_sync_if_needed(device)
        t1 = timer()
        time_records.append(TimeRecord(rank=rank, mode=mode, step=step, time_sec=t1 - t0))

        # 3. record memory after step
        mem_records.append(get_peak_record(rank, mode, "post_opt", device, model, optimizer))

    return mem_records, time_records

# --- ddp worker ---
def worker(
    rank: int, world_size: int, backend: str, master_addr: str, master_port: str, grad_comm_mode: str,
    vocab_size: int, context_length: int, global_batch_size: int, dtype_str: str,
    warmup_steps: int, measure_steps: int,
    out_dir: str,
):
    # setup process group
    setup_process_group(rank, world_size, backend, master_addr, master_port)

    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    all_mem_records: List[MemRecord] = []
    all_time_records: List[TimeRecord] = []

    modes = ["full", "sharded"]
    for mode in modes:
        mem_records, time_records = run_once(
            rank=rank, world_size=world_size, backend=backend, mode=mode, grad_comm_mode=grad_comm_mode,
            vocab_size=vocab_size, context_length=context_length, global_batch_size=global_batch_size, 
            dtype=dtype, warmup_steps=warmup_steps, measure_steps=measure_steps,
        )
        all_mem_records.extend(mem_records)
        all_time_records.extend(time_records)
        
        dist.barrier()
        torch.cuda.empty_cache()

    # gather results to rank 0
    gather_mem_records = [None for _ in range(world_size)]  # type: List[Any]
    gather_time_records = [None for _ in range(world_size)] # type: List[Any]

    dist.gather_object(all_mem_records, object_gather_list=gather_mem_records if rank == 0 else None, dst=0)
    dist.gather_object(all_time_records, object_gather_list=gather_time_records if rank == 0 else None, dst=0)

    if rank == 0:
        flat_mem_records: List[MemRecord] = [rec for per_rank in gather_mem_records for rec in per_rank]
        flat_time_records: List[TimeRecord] = [rec for per_rank in gather_time_records for rec in per_rank]

        df_mem_rec = pd.DataFrame([asdict(rec) for rec in flat_mem_records])
        df_time_rec = pd.DataFrame([asdict(rec) for rec in flat_time_records])

        def sumarize_stage(stage: str) -> pd.DataFrame:
            df_stage = df_mem_rec[df_mem_rec["stage"] == stage]
            df_summary = df_stage.groupby(["mode", "stage"], as_index=False)[
                ["params_MiB", "opt_MiB", "grads_MiB", "peak_alloc_MiB", "peak_reserved_MiB", "residual_MiB"]
            ].mean()
            return df_summary
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs(out_dir, exist_ok=True) 
        out_path_mem = Path(out_dir) / f"optimizer_state_sharding_mem_rank_{world_size}_{timestamp}.md"
        out_path_time = Path(out_dir) / f"optimizer_state_sharding_time_rank_{world_size}_{timestamp}.md"
        with open(out_path_mem, "w") as f_mem:
            f_mem.write("# Optimizer State Sharding Memory Benchmark\n\n")
            f_mem.write("## Detailed Records\n\n")
            f_mem.write(df_mem_rec.to_markdown(index=False))
            f_mem.write("\n\n")
            f_mem.write("## Summary by Stage\n\n")
            for stage in ["after_init", "pre_opt", "post_opt"]:
                f_mem.write(f"### Stage: {stage}\n\n")
                df_summary = sumarize_stage(stage)
                f_mem.write(df_summary.to_markdown(index=False))
                f_mem.write("\n\n")
        with open(out_path_time, "w") as f_time:
            f_time.write("# Optimizer State Sharding Time Benchmark\n\n")
            f_time.write("## Detailed Records\n\n")
            f_time.write(df_time_rec.to_markdown(index=False))
            f_time.write("\n\n")
            f_time.write("## Summary\n\n")
            df_time_summary = df_time_rec.groupby(["mode"], as_index=False)["time_sec"].mean()
            f_time.write(df_time_summary.to_markdown(index=False))
            f_time.write("\n\n")
        
    # teardown process group
    dist.barrier()
    teardown_process_group()

def main():
    # configuration
    world_size = 2
    backend = "nccl" # "gloo" for CPU, "nccl" for GPU
    master_addr = "localhost"
    master_port = "12355"

    vocab_size = 10000
    context_length = 256
    global_batch_size = 8
    dtype_str = "fp16"  # "fp16", "bf16", "fp32"
    grad_comm_mode = "" # "none", "bucketed", "indiv_overlap"; bucketed, indiv_overlap: OOM for large models

    warmpup_steps = 5
    measure_steps = 10
    out_dir = "results/optimizer_state_sharding_accounting"

    # spawn processes
    mp.spawn(
        worker,
        args=(
            world_size, backend, master_addr, master_port, grad_comm_mode,
            vocab_size, context_length, global_batch_size, dtype_str,
            warmpup_steps, measure_steps,
            out_dir,
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()

            












