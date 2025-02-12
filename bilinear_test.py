# Verify numerically that bilinear upscaling includes implicit replication padding

image_size = 48
scale_factor = 4

print("Max abs error (for a random input) between explicit and implicit repl padding in bilinear 4x upscale:")

import jax
jax.config.update("jax_enable_x64", True)
x = jax.random.uniform(jax.random.PRNGKey(42), (image_size, image_size), dtype=jax.numpy.float64)
x_padded = jax.numpy.pad(x, ((1, 1), (1, 1)), mode='edge')
x_resized_implicit = jax.image.resize(x, (x.shape[0]*scale_factor, x.shape[1]*scale_factor), "bilinear")
x_resized_explicit = jax.image.resize(x_padded, (x_padded.shape[0]*scale_factor, x_padded.shape[1]*scale_factor), "bilinear")[scale_factor:-scale_factor,scale_factor:-scale_factor]
print(f"JAX: {jax.numpy.max(jax.numpy.abs(x_resized_implicit - x_resized_explicit))}")

import torch
torch.manual_seed(42)
x = torch.rand(1, 1, 4, 4, dtype=torch.float64)
x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
x_resized_implicit = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')
x_resized_explicit = torch.nn.functional.interpolate(x_padded, scale_factor=scale_factor, mode='bilinear')[:,:,scale_factor:-scale_factor,scale_factor:-scale_factor]
print(f"PyTorch: {torch.max(torch.abs(x_resized_implicit - x_resized_explicit))}")

import cv2
import numpy as np
x = np.random.rand(image_size, image_size).astype(np.float64)
x_resized_implicit = cv2.resize(x, (x.shape[1] * scale_factor, x.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
x_padded = cv2.copyMakeBorder(x, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
x_resized_explicit = cv2.resize(x_padded, (x_padded.shape[1] * scale_factor, x_padded.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)[scale_factor:-scale_factor, scale_factor:-scale_factor]
print(f"OpenCV: {np.max(np.abs(x_resized_implicit - x_resized_explicit))}")