import torch
import torch.nn as nn
import itertools
from einops import rearrange, einsum
from tabulate import tabulate
from pathlib import Path
import pandas as pd
from datetime import datetime

from cs336_systems.FlashAttention_v2 import FlashAttention2Forward_triton
from cs336_basics.model import scaled_dot_product_attention
import triton.testing as tt

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = "cuda:3"

batch_size = 1

def make_inputs(seq_len, d_model, dtype):
    device = "cuda:3"
    Q = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
    K = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
    V = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype, requires_grad=True)
    dO = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)
    return Q, K, V, dO

def pytorch_attention(Q, K, V, is_causal):
    if is_causal:
        mask = torch.tril(torch.ones((Q.size(1), K.size(1)), device=Q.device)).bool()
    else:
        mask = None
    return scaled_dot_product_attention(Q, K, V, mask=mask)

def flash_attention(Q, K, V, is_causal):
    return FlashAttention2Forward_triton.apply(Q, K, V, is_causal)

def zero_grads(Q, K, V):
    Q.grad = None
    K.grad = None
    V.grad = None

def benchmark_one_case(seq_len, d_model, dtype):

    Q, K, V, dO = make_inputs(seq_len, d_model, dtype)

    is_causal = True

    # --- Pytorch ---
    def pytorch_fwd():
        return pytorch_attention(Q, K, V, is_causal)
    
    def pytorch_bwd():
        zero_grads(Q, K, V)
        out = pytorch_attention(Q, K, V, is_causal)
        out.backward(dO)

    def pytorch_fwd_bwd():
        zero_grads(Q, K, V)
        out = pytorch_attention(Q, K, V, is_causal)
        out.backward(dO)

    # --- FlashAttention-v2 Triton ---
    def flash_fwd():
        return flash_attention(Q, K, V, is_causal)
    
    def flash_bwd():
        zero_grads(Q, K, V)
        out = flash_attention(Q, K, V, is_causal)
        out.backward(dO)

    def flash_fwd_bwd():
        zero_grads(Q, K, V)
        out = flash_attention(Q, K, V, is_causal)
        out.backward(dO)

    # Warm-up
    for _ in range(5):
        pytorch_fwd_bwd()
        flash_fwd_bwd()
    torch.cuda.synchronize()

    # Benchmark
    pytorch_fwd_time = tt.do_bench(pytorch_fwd) # ms
    torch.cuda.synchronize()
    pytorch_bwd_time = tt.do_bench(pytorch_bwd) # ms
    torch.cuda.synchronize()
    pytorch_fwd_bwd_time = tt.do_bench(pytorch_fwd_bwd) # ms
    torch.cuda.synchronize()

    flash_fwd_time = tt.do_bench(flash_fwd) # ms
    torch.cuda.synchronize()
    flash_bwd_time = tt.do_bench(flash_bwd) # ms
    torch.cuda.synchronize()
    flash_fwd_bwd_time = tt.do_bench(flash_fwd_bwd) # ms
    torch.cuda.synchronize()

    return {
        "seq": seq_len,
        "d_model": d_model,
        "dtype": str(dtype).split(".")[-1],
        "pytorch_fwd": pytorch_fwd_time,
        "pytorch_bwd": pytorch_bwd_time,
        "pytorch_fwd_bwd": pytorch_fwd_bwd_time,
        "flash_fwd": flash_fwd_time,
        "flash_bwd": flash_bwd_time,
        "flash_fwd_bwd": flash_fwd_bwd_time,
    }

def main():
    seq_lens = [2 ** k for k in range(7, 17)]  # 128 to 65536
    d_models = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]

    results_row = []
    for seq, d_model, dt in itertools.product(seq_lens, d_models, dtypes):
        print(f"Running benchmark for seq_len={seq}, d_model={d_model}, dtype={dt}")
        try:
            res = benchmark_one_case(seq, d_model, dt)
            results_row.append(res)

        except Exception as e:
            if 'out of memory' in str(e):
                print(f"OOM for seq_len = {seq}, d_model={d_model}, dtype={dt}")
                torch.cuda.empty_cache()
                results_row.append({
                    "seq": seq,
                    "d_model": d_model,
                    "dtype": str(dt).split(".")[-1],
                    "pytorch_fwd": "OOM",
                    "pytorch_bwd": None,
                    "pytorch_fwd_bwd": None,
                    "flash_fwd": "OOM",
                    "flash_bwd": None,
                    "flash_fwd_bwd": None
                })
            else:
                raise e
        # del Q, K, V, dO
        torch.cuda.empty_cache()
    results_df = pd.DataFrame(results_row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"results/flash_benchmark/flash_benchmark_results_{timestamp}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)

    output_path = Path(f"results/flash_benchmark/flash_benchmark_results_{timestamp}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(f"results/flash_benchmark/flash_benchmark_{timestamp}.md", "w") as f:
            f.write(results_df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    torch.cuda.set_device(3)
    main()
