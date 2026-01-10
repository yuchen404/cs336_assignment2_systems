import submitit
import itertools
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime

model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Output_file = f"results/benchmark_results_{timestamp}.csv"

def run_one_job(config):
    cmd = [
        "python", "cs336_systems/benchmarking_script.py",
        "--model_size", config["model_size"],
        "--num_layers", str(config["num_layers"]),
        "--d_model", str(config["d_model"]),
        "--d_ff", str(config["d_ff"]),
        "--num_heads", str(config["num_heads"]),
        "--context_length", str(config["context_length"]),
        "--batch_size", str(config["batch_size"]),
        "--mode", config["mode"],
        "--precision", config["precision"],
        "--output_file", Output_file
    ]
    subprocess.run(cmd, check=True)


# ========= Submitit job submission =========
def main():
    # executor = submitit.AutoExecutor(folder="slurm_logs")
    # executor.update_parameters(
    #     mem_gb=32,
    #     gpus_per_node=1,
    #     cpus_per_task=4,
    #     timeout_min=30,
    #     slurm_partition="dev",
    #     slurm_array_parallelism=1
    # )

    context_lengths = [128, 256, 512, 1024]
    # dtypes = [ "fp16", "bf16"]
    dtypes = ["fp16"]
    modes = ["forward_only", "forward_backward"]
    # modes = ["forward_only"]
    batch = 4

    jobs = []

    for model_size, model_params in model_configs.items():
        for context_length, dtype, mode in itertools.product(
            context_lengths, dtypes, modes
        ):
            config = {
                "model_size": model_size,
                "num_layers": model_params["num_layers"],
                "d_model": model_params["d_model"],
                "d_ff": model_params["d_ff"],
                "num_heads": model_params["num_heads"],
                "context_length": context_length,
                "batch_size": batch,
                "mode": mode,
                "precision": dtype,
            }
            jobs.append(config)

    # Path("slurm_logs").mkdir(exist_ok=True)
    print(f"Submitting {len(jobs)} benchmark jobs...")
    for cfg in jobs:
        if cfg["model_size"] == "large" and cfg["context_length"] >= 1024:
            # print("Skipping large with context length 1024 due to resource constraints.")
            continue
        if cfg["model_size"] == "xl" and cfg["context_length"] >= 512:
            # print("Skipping xl with context length 512 due to resource constraints.")
            continue
        if cfg["model_size"] == "2.7B" and cfg["context_length"] >= 256:
            # print("Skipping 2.7B with context length 256 due to resource constraints.")
            continue
        run_one_job(cfg)

    df = pd.read_csv(f"results/benchmark_results_{timestamp}.csv")
    # df.to_markdown(
    #     index=False,
    #     floatfmt=".4f"
    # )
    
    output_path = Path(f"results/benchmark_results_{timestamp}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"results/benchmark_results_{timestamp}.md", "w") as f:
        f.write(df.to_markdown(
            index=False,
            floatfmt=".4f"
        ))

    # executor.map_array(run_one_job, jobs)
    # print(f"Submitted {len(jobs)} benchmark jobs.")

if __name__ == "__main__":
    main()