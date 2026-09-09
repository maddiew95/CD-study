from sklearn.preprocessing import StandardScaler
import numpy as np, torch, time

COLS = slice(1, 14)
LABEL = -1

def count_parameters(model):
    """Count trainable parameters. Returns (total, by_type_dict)."""
    total = 0
    by_type = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            kind = name.split(".")[0]        # e.g. 'conv', 'lstm', 'fc'
            by_type[kind] = by_type.get(kind, 0) + n
    return total, by_type


def reset_gpu_peak_memory(device):
    """Call before training to get a clean peak-memory measurement."""
    if "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_peak_memory_mb(device):
    """Returns peak CUDA memory in MB since last reset. 0 if on CPU."""
    if "cuda" in str(device):
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


class Timer:
    """Context manager for wall-clock timing with CUDA sync."""
    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device is not None and "cuda" in str(self.device):
            torch.cuda.synchronize(self.device)
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        if self.device is not None and "cuda" in str(self.device):
            torch.cuda.synchronize(self.device)
        self.elapsed = time.time() - self.t0


def measure_inference_time(predict_fn, X_test, device, n_warmup=5, n_runs=20):
    """
    Measure mean inference time per sample.
    predict_fn: a callable that takes X (tensor) and returns predictions.
    """
    # warmup runs -- first few calls are always slow due to kernel compilation
    for _ in range(n_warmup):
        _ = predict_fn(X_test)

    # actual measurement
    times = []
    for _ in range(n_runs):
        with Timer(device) as t:
            _ = predict_fn(X_test)
        times.append(t.elapsed)

    mean_total = sum(times) / len(times)
    return {
        "total_sec":     mean_total,
        "per_sample_ms": 1000 * mean_total / len(X_test),
        "n_samples":     len(X_test),
    }