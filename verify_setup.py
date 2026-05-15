import sys

print(f"Python: {sys.version}")
print(f"Path:   {sys.executable}")

# Test MLX
import mlx.core as mx
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([4.0, 5.0, 6.0])
z = mx.add(x, y)
mx.eval(z)
print(f"MLX:    {x} + {y} = {z}")

# Test NumPy
import numpy as np
print(f"NumPy:  {np.__version__}")

# Test PIL
from PIL import Image
print(f"Pillow: OK")

# Test huggingface
import huggingface_hub
print(f"HF Hub: {huggingface_hub.__version__}")

print("\n=== All good! Ready to run sparse_edit. ===")
