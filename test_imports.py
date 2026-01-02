print("Importing torch...")
import torch
print(f"Torch imported: {torch.__version__}")

print("Importing numpy...")
import numpy as np

print("Importing pandas...")
import pandas as pd

print("Importing matplotlib...")
import matplotlib.pyplot as plt
print("All imports successful.")

if torch.cuda.is_available():
    print("Initializing CUDA...")
    torch.cuda.init()
    print(f"CUDA Initialized. Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available.")
