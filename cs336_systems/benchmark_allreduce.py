import argparse
from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
from datetime import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"    # Adjust as needed

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Tuple, List, Any


# --- Benchmarking Utilities ---
def bytes_from_mb(mb: int) -> int:
    return mb * 1024 * 1024

def recommend_iters(bytes_size: int, user_iters: int | None) -> int:
    if user_iters is not None:
        return user_iters
    
    if bytes_size <= bytes_from_mb(1): return 200
    if bytes_size <= bytes_from_mb(10): return 100
    if bytes_size <= bytes_from_mb(100): return 20
    # 1GB+
    return 5

# --- Generate random tensor ---
def make_tensor(bytes_size: int, device: torch.device) -> torch.Tensor:
    numel = bytes_size // 4  # float32
    x = torch.randn((numel,), dtype = torch.float32, device=device) 
    return x

# --- CUDA Synchronization Utility ---
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
# Benchmark Functions
# ===================

def benchmark_allreduce_once(input_data: torch.Tensor, warmup: int, iters: int) -> Dict[str, float]:
    """
    Benchmark a single all-reduce operation on the given tensor.
    Returns per-rank timing statistics.
    """
    device = input_data.device

    # all ranks start together
    dist.barrier()

    # Warm-up
    for _ in range(warmup):
        dist.all_reduce(input_data, async_op=False)
    cuda_sync_if_needed(device)

    # Measuring time
    times = []
    for _ in range(iters):
        cuda_sync_if_needed(device)

        t0 = timer()
        dist.all_reduce(input_data, async_op=False)

        cuda_sync_if_needed(device)
        t1 = timer()
        times.append(t1 - t0)

    avg_time = sum(times) / iters 
    p50_time = sorted(times)[iters // 2] 
    min_time = min(times) 
    max_time = max(times) 
    return {
        "avg_time_sec": avg_time,
        "p50_time_sec": p50_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
    }

def worker(rank: int, world_size: int, backend: str, master_addr: str, master_port: str, sizes_mb: List[int], warmup: int, iters: int | None) -> None:
    """
    One worker process for benchmarking all-reduce.
    NCLL 
    Gloo
    """
    # setup 
    setup_process_group(rank, world_size, backend, master_addr, master_port)

    try:
        if backend == "nccl":
            # each process uses its own GPU
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        # Collect results
        per_rank_results: List[Dict[str, Any]] = []

        for mb in sizes_mb:
            bytes_size = bytes_from_mb(mb)
            if bytes_size % 4 != 0:
                bytes_size = (bytes_size // 4) * 4  # align to float32

            local_iters = recommend_iters(bytes_size, iters)

            # Create input tensor
            input_x = make_tensor(bytes_size, device)

            # Benchmark once
            res_sta = benchmark_allreduce_once(input_x, warmup, local_iters)

            effective_gbps = (bytes_size / res_sta["avg_time_sec"]) / (1024 ** 3)

            per_rank_results.append({
                "rank": rank,
                "size_mb": mb,
                "world_size": world_size,
                "backend": backend,
                "device": str(device),
                "iters": local_iters,
                **res_sta,
                "effective_gbps": effective_gbps,
            })
        
        # Gather results at rank 0
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, per_rank_results)

        if rank == 0:
            all_records = [rec for rank_res in gathered_results for rec in rank_res]    # type: ignore

            rows_by_size: Dict[int, List[Dict[str, Any]]] = {}
            for rec in all_records:
                sz = rec["size_mb"]
                if sz not in rows_by_size:
                    rows_by_size[sz] = []
                rows_by_size[sz].append(rec)
            # Aggregate per size
            aggregated_results: List[Dict[str, Any]] = []
            for sz, recs in rows_by_size.items():
                aggregated_results.append({
                    "size_mb": sz,
                    "world_size": world_size,
                    "backend": backend,
                    "device": recs[0]["device"],
                    "avg_time_sec": sum(r["avg_time_sec"] for r in recs) / len(recs),
                    "p50_time_sec": sum(r["p50_time_sec"] for r in recs) / len(recs),
                    "min_time_sec": min(r["min_time_sec"] for r in recs),
                    "max_time_sec": max(r["max_time_sec"] for r in recs),
                    "effective_gbps": sum(r["effective_gbps"] for r in recs) / len(recs),
                })
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_rank_csv = f"results/benchmark_allreduce/allreduce_benchmark_rank_{backend}_ws{world_size}_{timestamp}.csv"
            output_rank_md = f"results/benchmark_allreduce/allreduce_benchmark_rank_{backend}_ws{world_size}_{timestamp}.md"
            write_csv(output_rank_csv, output_rank_md, all_records)
            # Save aggregated
            output_agg_csv = f"results/benchmark_allreduce/allreduce_benchmark_agg_{backend}_ws{world_size}_{timestamp}.csv"
            output_agg_md = f"results/benchmark_allreduce/allreduce_benchmark_agg_{backend}_ws{world_size}_{timestamp}.md"
            write_csv(output_agg_csv, output_agg_md, aggregated_results)

    finally:
        teardown_process_group()

def write_csv(path_csv: str, path_md: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.sort_values(by=["size_mb", "world_size", "backend"], inplace=True)
    df.to_csv(path_csv, index=False)
    print(f"Saved aggregated results to {path_csv}")

    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w") as f:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark All-Reduce Operation")
    parser.add_argument("--world_size", type=int, default=6, help="Number of processes (world size)")
    parser.add_argument("--backend", type=str, choices=["gloo", "nccl"], default="gloo", help="Distributed backend to use")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed setup")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port for distributed setup")
    parser.add_argument("--sizes_mb", type=int, nargs="+", default=[1, 10, 100, 1000], help="List of tensor sizes in MB to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up iterations")
    parser.add_argument("--iters", type=int, default=None, help="Number of benchmark iterations (auto if not set)")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.backend == "nccl":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # Adjust as needed

    mp.set_start_method("spawn", force=True)

    mp.spawn(fn=worker, args=(args.world_size, args.backend, args.master_addr, args.master_port, args.sizes_mb, args.warmup, args.iters), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()

    