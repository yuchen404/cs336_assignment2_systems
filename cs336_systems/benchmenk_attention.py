import torch
import torch.nn as nn
from timeit import default_timer as timer
import pandas as pd
import torch.cuda.nvtx as nvtx
import math
from cs336_basics.model import CausalMultiHeadSelfAttention
from datetime import datetime
from pathlib import Path


device = "cuda:3"
dtype = torch.float32

Batch_Size = 8
Warmup_Steps = 10
Measure_Steps = 100

d_models = [16, 32, 64, 128]
seq_lengths = [256, 1024, 4096, 8192, 16384]

class AttentionModule(nn.Module):
    def forward(self, Q, K, V):
        d = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V

results = []

attention_module = AttentionModule().to(device=device, dtype=dtype)
compiled_attention = torch.compile(attention_module, mode="default")

modules = {
    "vanilla": attention_module,
    # "compiled": compiled_attention
}


for d in d_models:
    for seq_len in seq_lengths:
        print(f"Running attention with d_model={d}, seq_length={seq_len}")
        torch.cuda.reset_peak_memory_stats()
        try:

            # attention = attention.to(device)

            for module_name, att_module in modules.items():
                torch.cuda.reset_peak_memory_stats(device=device)
                Q = torch.randn(Batch_Size, seq_len, d, device=device, dtype=dtype)
                K = torch.randn(Batch_Size, seq_len, d, device=device, dtype=dtype)
                V = torch.randn(Batch_Size, seq_len, d, device=device, dtype=dtype)

                Q.requires_grad_(True)
                K.requires_grad_(True)
                V.requires_grad_(True)

                # Warm-up
                for _ in range(Warmup_Steps):
                    
                    out = att_module(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                torch.cuda.synchronize()
                fwd_time = 0.0
                bwd_time = 0.0
                # Measure forward time
                # torch.cuda.reset_peak_memory_stats(device)
                start_time = timer()
                for ite in range(Measure_Steps):
                    # torch.cuda.reset_peak_memory_stats(device=device)
                    start_time_fwd = timer()
                    out = att_module(Q, K, V)
                    torch.cuda.synchronize()
                    fwd_time += (timer() - start_time_fwd) 

                    # Measure Memory usage
                    if ite == 0:
                        mem_before_bwd = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3) # in GB
                        
                        # mem_before_bwd = torch.cuda.memory_allocated(device=device) / (1024 ** 2) # in MB
                        torch.cuda.reset_peak_memory_stats(device=device)

                    # Measure backward time
                        
                    start_time_bwd = timer()
                    loss = out.sum()
                    loss.backward()
                    torch.cuda.synchronize()
                    bwd_time += (timer() - start_time_bwd) 
                    if ite == 0:
                        mem_after_bwd = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3) # in GB
                        # mem_after_bwd = torch.cuda.memory_allocated(device=device) / (1024 ** 2) # in MB
                    
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                fwd_time = fwd_time / Measure_Steps * 1000 # in ms
                bwd_time = bwd_time / Measure_Steps * 1000 # in ms
                

                results.append({
                    "module": module_name,
                    "d_model": d,
                    "seq_length": seq_len,
                    "fwd_time_ms": fwd_time,
                    "bwd_time_ms": bwd_time,
                    "memory_before_bwd_GB": mem_before_bwd,
                    "memory_after_bwd_GB": mem_after_bwd,
                    "status": "normal"
                })
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM for d_model={d}, seq_length={seq_len}")
                torch.cuda.empty_cache()
                results.append({
                    "module": module_name,
                    "d_model": d,
                    "seq_length": seq_len,
                    "fwd_time_ms": None,
                    "bwd_time_ms": None,
                    "memory_before_bwd_GB": None,
                    "memory_after_bwd_GB": None,
                    "status": "OOM"
                })
            else:
                raise e
        del Q, K, V
        torch.cuda.empty_cache()

df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = Path(f"results/benchmark_results_{timestamp}.md")
# output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(f"results/attention_benchmark_results_{timestamp}.csv", index=False)
with open(f"results/attention_benchmark_{timestamp}.md", "w") as f:
        f.write(df.to_markdown(
            index=False,
            floatfmt=".4f"
        ))
    