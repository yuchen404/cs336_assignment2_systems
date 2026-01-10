from cs336_basics.model import scaled_dot_product_attention
import torch
from torch.profiler import profile, record_function

batch_size = 8
d_models = [16, 32, 64, 128]
# seq_lens = [256, 1024, 4096, 8192, 16384]
seq_lens = [256, 1024, 4096, 8192]

for d_model in d_models:
    for seq_len in seq_lens:
        print(f"Profiling d_model={d_model}, seq_len={seq_len}")
        Q, K, V = torch.randn(
            (3, batch_size, seq_len, d_model), requires_grad=True).cuda()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(10):
                if i % 20 == 0:
                    print(f"  Progress: {i}/102")

                with record_function("forward"):
                    logits = scaled_dot_product_attention(Q, K, V)

                loss = logits.sum()

                with record_function("backward"):
                    loss.backward()

                Q.grad = None
                K.grad = None
                V.grad = None

                prof.step()

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
