# from __future__ import annotations
import argparse
import torch
import timeit
import numpy as np
import pandas as pd
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
# from annotated import annotated_scaled_dot_product_attention
from cs336_basics import model as cs336_model
from contextlib import nullcontext
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking script basics")

    # --- model ---
    parser.add_argument("--model_size", type=str, default="medium")
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--context_length", type=int, default=256)
    # --- execution ---
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", default="forward_only", choices=["forward_only", "forward_backward"])

    # --- precision ---
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")

    # --- output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output_file", type=str, default=f"results/benchmark_results_{timestamp}.csv")

    return parser.parse_args()

def benchmarking():
    args = parse_args()

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device = "cuda:3"

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    dtype = dtype_map[args.precision]

    torch.manual_seed(42)

    warmup_steps = 5
    measure_steps = 10
    vocab_size = 10000

    batch = args.batch_size
    context_length = args.context_length
    d_model = args.d_model
    num_heads = args.num_heads
    d_ff = args.d_ff
    num_layers = args.num_layers
    mode = args.mode

    # --- initialize model ---
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    )
    model.to(device=device, dtype=dtype)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.train()

    # --- random input --- 
    data_random = np.random.randint(0, vocab_size, (1 << 12, ))
    input_x, target_y = get_batch(
        dataset=data_random,
        batch_size=batch,
        context_length=context_length,
        device=device,
    )
    # target_y = torch.randint(0, vocab_size, (batch, context_length), device=device)
    precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.precision != "default" else nullcontext()

    # --- warmup ---
    for _ in range(warmup_steps):
        with precision_context:
            if mode == "forward_only":
                logits = model(input_x)
            
            elif mode == "forward_backward":
                logits = model(input_x)
                loss = cross_entropy(logits, target_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

    # --- measurement ---
    times = []
    fwd_times = []
    bwd_times = []
    for _ in range(measure_steps):

        torch.cuda.synchronize()
        start_time = timer()
        with precision_context:
            if mode == "forward_only":
                logits = model(input_x)

            elif mode == "forward_backward":
                # forward
                logits = model(input_x)
                loss = cross_entropy(logits, target_y)

                torch.cuda.synchronize()
                t1 = timer()

                # backward
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()
        end_time = timer()
        time_ms = (end_time - start_time) * 1000

        if mode == "forward_backward":
            bwd_times.append((end_time - t1) * 1000)
            fwd_times.append((t1 - start_time) * 1000)

        times.append(time_ms)

    times = np.array(times)
    if mode == "forward_backward":
        bwd_times = np.array(bwd_times)
        fwd_times = np.array(fwd_times)
        avg_fwd_time = np.mean(fwd_times)
        std_fwd_time = np.std(fwd_times)
        avg_bwd_time = np.mean(bwd_times)
        std_bwd_time = np.std(bwd_times)

    avg_time = np.mean(times)
    std_time = np.std(times)

    # --- output results ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "forward_only":
        row = {
            "size": args.model_size,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "context_length": context_length,
            "batch_size": batch,
            "precision": args.precision,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "mode": mode,
        }
    elif mode == "forward_backward":
        row = {
            "size": args.model_size,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "context_length": context_length,
            "batch_size": batch,
            "precision": args.precision,
            "avg_time_ms": avg_time,
            # "std_time_ms": std_time,
            "avg_fwd_time_ms": avg_fwd_time,
            # "std_fwd_time_ms": std_fwd_time,
            "avg_bwd_time_ms": avg_bwd_time,
            # "std_bwd_time_ms": std_bwd_time,
            "mode": mode,
        }

    df = pd.DataFrame([row])
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(output_path, index=False)


model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

if __name__ == "__main__":
    benchmarking()