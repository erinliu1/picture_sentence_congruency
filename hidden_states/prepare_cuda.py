# Sets up the environment variables PyTorch needs to find the right GPU and the right CUDA toolkit.
# Call prepare_cuda() as the very first thing you do, before torch gets imported anywhere in the script. 

# Sets up the environment variables PyTorch needs to find the right GPU and the right CUDA toolkit.
# Call prepare_cuda() as the very first thing you do, before torch gets imported anywhere in the script. 

def prepare_cuda(allow_multi_gpu=False):
    import os
    import subprocess
    from pathlib import Path

    # If using a single GPU, automatically choose the GPU
    # with the most currently available VRAM.
    if not allow_multi_gpu:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpus = []

        for line in result.stdout.strip().splitlines():
            index, free_memory, total_memory = [
                int(x.strip()) for x in line.split(",")
            ]

            gpus.append(
                {
                    "index": index,
                    "free_memory": free_memory,
                    "total_memory": total_memory,
                }
            )

        if not gpus:
            raise RuntimeError("No NVIDIA GPUs found.")

        # Pick GPU with the most free VRAM.
        best_gpu = max(gpus, key=lambda gpu: gpu["free_memory"])

        physical_gpu_index = best_gpu["index"]

        # Force this process to only see the selected GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)

        print(
            f"Selected GPU {physical_gpu_index}: "
            f"{best_gpu['free_memory']} MiB free / "
            f"{best_gpu['total_memory']} MiB total"
        )

    # CUDA toolkit configuration
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3")
    cuda_include = os.environ.get(
        "CUDA_INCLUDE_DIR",
        f"{cuda_home}/targets/sbsa-linux/include",
    )
    cuda_library = os.environ.get(
        "CUDA_LIBRARY_DIR",
        f"{cuda_home}/targets/sbsa-linux/lib",
    )
    ptxas_path = os.environ.get(
        "TRITON_PTXAS_PATH",
        f"{cuda_home}/bin/ptxas",
    )

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["TRITON_PTXAS_PATH"] = ptxas_path

    os.environ["PATH"] = os.pathsep.join(
        value
        for value in [
            str(Path(ptxas_path).parent),
            os.environ.get("PATH"),
        ]
        if value
    )

    os.environ["CPATH"] = os.pathsep.join(
        value
        for value in [
            cuda_include,
            os.environ.get("CPATH"),
        ]
        if value
    )

    os.environ["LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in [
            cuda_library,
            os.environ.get("LIBRARY_PATH"),
        ]
        if value
    )

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in [
            cuda_library,
            os.environ.get("LD_LIBRARY_PATH"),
        ]
        if value
    )