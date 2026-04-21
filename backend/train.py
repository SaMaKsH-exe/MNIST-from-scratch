"""Numpy-based prediction entry point used by the frontend API endpoint.

Replace `predict_digit_from_pixels` with your real neural network inference.
"""

from __future__ import annotations

import numpy as np


def predict_digit_from_pixels(pixels: list[int] | np.ndarray) -> tuple[int, float]:
	"""Return (digit, confidence) for a 784-length grayscale vector.

	The input is expected in MNIST-style ordering (row-major, 28x28),
	with values in 0..255.
	"""

	arr = np.asarray(pixels, dtype=np.float32)
	if arr.size != 784:
		raise ValueError(f"Expected 784 pixels, got {arr.size}")

	# Normalize to 0..1 and reshape for your model.
	arr = arr.reshape(28, 28) / 255.0

	# TODO: Replace this placeholder with your numpy neural net forward pass.
	# Example contract:
	#   probs = model_forward(arr.flatten())
	#   digit = int(np.argmax(probs))
	#   confidence = float(probs[digit])
	digit = int(np.clip(np.round(arr.mean() * 9), 0, 9))
	confidence = float(np.clip(arr.max(), 0.0, 1.0))

	return digit, confidence
