import torch

def gpu_acces():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")
    else:
        print("GPU not found. PyTorch will use the CPU.")
