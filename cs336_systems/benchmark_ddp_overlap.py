import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import argparse
from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass

import torch.cuda.nvtx as nvtx

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp  
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp_utils import (
    cuda_sync_if_needed, _broadcast_params,
    ddp_individual_parameters_on_after_backward,
    ddp_flattened_parameters_on_after_backward,
    )

from cs336_systems.ddp_overlap import DDPIndividualParameters, DDPBucketedParameters

# ===================
@dataclass
class StepRes:
    rank: int
    mode: str
    step: int
    total_s: float
    bwd_s: float
    sync_s: float   # for overlap: wait + divide time; for sync: comm time
    fwd_s: float

# ---- training step function ----
def train_once(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_x,
    target_y,
    device: torch.device,
    mode: str,  # "ddp_bucketed_overlap", "ddp_indiv_overlap", "ddp_indiv_sync", "ddp_flattened_sync"
) -> Tuple[float, float, float, float]:
    """
    Run one training step and return timing statistics.
    """
    model.train()
    cuda_sync_if_needed(device)
    t0 = timer()

    # --- forward ---
    time_f0 = timer()
    with nvtx.range("Forward"):
        logits = model(input_x)
        loss = cross_entropy(logits, target_y)
    # cuda_sync_if_needed(device)
    time_f1 = timer()

    # --- backward ---
    time_b0 = timer()
    with nvtx.range("Backward"):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    # cuda_sync_if_needed(device)
    time_b1 = timer()

    # --- DDP gradient synchronization ---
    time_comm0 = timer()
    with nvtx.range("DDP Sync"):
        if mode == "ddp_indiv_overlap":
            assert isinstance(model, DDPIndividualParameters)
            model.finish_gradient_synchronization()
            cuda_sync_if_needed(device)
        elif mode == "ddp_indiv_sync":
            ddp_individual_parameters_on_after_backward(model)
            cuda_sync_if_needed(device)
        elif mode == "ddp_flattened_sync":
            ddp_flattened_parameters_on_after_backward(model)
            cuda_sync_if_needed(device)
        elif mode == "ddp_bucketed_overlap":
            assert isinstance(model, DDPBucketedParameters)
            model.finish_gradient_synchronization()
            cuda_sync_if_needed(device)
            
    time_comm1 = timer()
    
    sync_time = time_comm1 - time_comm0

    # --- optimizer step ---
    with nvtx.range("Opt_Step"):
        optimizer.step()
    cuda_sync_if_needed(device)
    t1 = timer()

    total_time = t1 - t0
    fwd_time = time_f1 - time_f0
    bwd_time = time_b1 - time_b0

    return total_time, bwd_time, sync_time, fwd_time
    

# ---- Distributed stepup ----
def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def teardown_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    mode: str,
    vocab_size: int,
    context_length: int,
    global_batch_size: int,
    dtype_str: str,
    warmup_steps: int,
    measure_steps: int,
    out_dir: str,
    bucket_size_mb: float,
) -> None:
    """
    one worker process for benchmarking DDP with different modes
    """
    setup_process_group(rank, world_size, backend, master_addr, master_port)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    device_str = f"cuda:{rank}" if backend == "nccl" else "cpu"

    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    assert global_batch_size % world_size == 0, "Global batch size must be divisible by world size"
    local_batch_size = global_batch_size // world_size

    # --- model init ---
    torch.manual_seed(42)
    base_model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    ).to(device=device, dtype=dtype)

    if mode == "ddp_indiv_overlap":
        model = DDPIndividualParameters(base_model)
    elif mode == "ddp_bucketed_overlap":
        model = DDPBucketedParameters(base_model, bucket_size_mb)
    else:
        _broadcast_params(base_model, src=0)
        model = base_model

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # --- synthetic data ---
    g = torch.Generator().manual_seed(42 + rank)
    data_random = torch.randint(0, vocab_size, (1 << 10, ), generator=g).numpy()

    # --- warm-up ---
    for step in range(warmup_steps):
        input_x, target_y = get_batch(data_random, local_batch_size, context_length, device_str)
        _ = train_once(model, optimizer, input_x, target_y, device, mode)
    dist.barrier()

    # --- measurement ---
    step_results: List[StepRes] = []
    for step in range(measure_steps):
        input_x, target_y = get_batch(data_random, local_batch_size, context_length, device_str)
        with nvtx.range(f"Rank {rank} Step {step}"):
            total_s, bwd_s, sync_s, fwd_s = train_once(model, optimizer, input_x, target_y, device, mode)
        step_results.append(StepRes(
            rank=rank,
            mode=mode,
            step=step,
            total_s=total_s,
            bwd_s=bwd_s,
            sync_s=sync_s,
            fwd_s=fwd_s,
        ))
    
    # gather results to rank 0
    all_results = [None for _ in range(world_size)]  # type: List[Any]
    dist.gather_object(step_results, object_gather_list=all_results if rank == 0 else None, dst=0)

    if rank == 0:
        flat_results: List[StepRes] = [rec for rank_res in all_results for rec in rank_res]
        per_step_records: List[Dict[str, Any]] = []
        for step in range(measure_steps):
            rec_by_rank = [r for r in flat_results if r.step == step]
            for r in rec_by_rank:
                per_step_records.append({
                    "mode": r.mode,
                    "rank": r.rank,
                    "step": r.step,
                    "total_s_avg": sum(r.total_s for r in rec_by_rank) / len(rec_by_rank),
                    "fwd_s_avg": sum(r.fwd_s for r in rec_by_rank) / len(rec_by_rank),
                    "bwd_s_avg": sum(r.bwd_s for r in rec_by_rank) / len(rec_by_rank),
                    "sync_s_avg": sum(r.sync_s for r in rec_by_rank) / len(rec_by_rank),
                })
        
        totals = [rec["total_s_avg"] for rec in per_step_records]
        syncs = [rec["sync_s_avg"] for rec in per_step_records]
        fwds = [rec["fwd_s_avg"] for rec in per_step_records]
        bwds = [rec["bwd_s_avg"] for rec in per_step_records]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        summary = {
            "date": timestamp,
            "mode": mode,
            "backend": backend,
            "world_size": world_size,
            # "global_batch_size": global_batch_size,
            "vocab_size": vocab_size,
            "d_model": 1600,
            "context_length": context_length,
            "dtype": dtype_str,
            "bucket_size_mb": bucket_size_mb if "bucketed" in mode else "N/A",
            "total_s_avg": sum(totals) / len(totals),
            "fwd_s_avg": sum(fwds) / len(fwds),
            "bwd_s_avg": sum(bwds) / len(bwds),
            "sync_s_avg": sum(syncs) / len(syncs),
        }

        os.makedirs(out_dir, exist_ok=True)
        per_step_file = os.path.join(out_dir, f"per_step_{mode}_ws{world_size}_{timestamp}.md")
        summary_file = os.path.join(out_dir, f"summary_ws{world_size}.md")
        with open(per_step_file, "w") as f_md:
            df_steps = pd.DataFrame(per_step_records)
            f_md.write(df_steps.to_markdown(index=False, floatfmt=".4f"))
        with open(summary_file, "a") as f_md:
            df_summary = pd.DataFrame([summary])
            f_md.write("\n" + df_summary.to_markdown(index=False, floatfmt=".4f") + "\n")
        print(f"Saved per-step results to {per_step_file}")
        print(f"Saved summary results to {summary_file}")
    
    dist.barrier()
    # --- teardown process group ---
    teardown_process_group()

# ===================
# Main
# ===================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP with individual parameter overlap")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes / GPUs")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend (nccl or gloo)")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed setup")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port for distributed setup")
    parser.add_argument("--mode", type=str, choices=["ddp_bucketed_overlap", "ddp_indiv_overlap", "ddp_indiv_sync", "ddp_flattened_sync"], default="ddp_bucketed_overlap", help="DDP mode to benchmark")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    parser.add_argument("--global_batch_size", type=int, default=8, help="Global batch size")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16", help="Data type for model parameters")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--measure_steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--out_dir", type=str, default="results/benchmark_ddp_overlap", help="Output directory for results")
    parser.add_argument("--bucket_size_mb", type=float, default=100.0, help="Bucket size in MB for bucketed DDP")
    return parser.parse_args()

def main(): 
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(  # type: ignore
        worker,
        args=(
            args.world_size,
            args.backend,
            args.master_addr,
            args.master_port,
            args.mode,
            args.vocab_size,
            args.context_length,
            args.global_batch_size,
            args.dtype,
            args.warmup_steps,
            args.measure_steps,
            args.out_dir,
            args.bucket_size_mb,
        ),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
    
