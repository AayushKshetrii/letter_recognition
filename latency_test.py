"""
Latency measurement for the LetterNet model.
Run from inside the letter_train/ directory:
    python latency_test.py
"""
import time
import torch
import json

# Reuse the architecture from main.py
from main import LetterNet, LABELS, NUM_CLASSES, MEAN, STD  # noqa

MODEL_PATH = "model_final.pth"

model = LetterNet(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Dummy 1x1x28x28 input (same shape the API receives)
test_tensor = torch.randn(1, 1, 28, 28)

# Warm-up (first run pays JIT/setup cost we don't want in the average)
with torch.no_grad():
    for _ in range(10):
        _ = model(test_tensor)

# Measure
times_ms = []
with torch.no_grad():
    for _ in range(100):
        start = time.perf_counter()
        _ = model(test_tensor)
        times_ms.append((time.perf_counter() - start) * 1000)

avg = sum(times_ms) / len(times_ms)
p50 = sorted(times_ms)[len(times_ms) // 2]
p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
mn = min(times_ms)
mx = max(times_ms)

print(f"Device: CPU")
print(f"Runs:   {len(times_ms)}")
print(f"Avg:    {avg:.1f} ms")
print(f"p50:    {p50:.1f} ms")
print(f"p95:    {p95:.1f} ms")
print(f"Min:    {mn:.1f} ms")
print(f"Max:    {mx:.1f} ms")
