import torch
import time

# Choose a specific GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example workload: matrix multiplication
size = 1024  # safe size to avoid crashing GPU
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

print("Starting safe infinite loop on allocated GPU... Press Ctrl+C to stop.")

try:
    while True:
        c = torch.matmul(a, b)  # heavy computation
        time.sleep(0.1)         # slight pause to reduce total load
except KeyboardInterrupt:
    print("Stopped safely.")
