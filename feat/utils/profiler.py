"""
FEAT Performance Profiler
=========================
Lightweight GPU time + memory profiler for diagnosing inference bottlenecks.

Usage:
    from feat.utils.profiler import FEATProfiler

    profiler = FEATProfiler(enabled=True, device='cuda')
    profiler.start("forward_pass")

    profiler.mark("x_preprocess")
    # ... do x preprocessing ...
    profiler.mark("encoder_x")
    # ... do x encoding ...
    profiler.mark("done")

    profiler.stop()
    profiler.summary()
"""

import time
import torch
from collections import OrderedDict


class FEATProfiler:
    """
    Tracks wall-clock time and GPU memory between named checkpoints.

    Each mark() call records the time and GPU memory since the last mark.
    Results are stored per-stage and can be printed as a summary table.

    Thread-safe for single-GPU inference (not designed for multi-GPU DDP).
    """

    def __init__(self, enabled: bool = False, device: str = 'cuda', sync_cuda: bool = True):
        """
        Args:
            enabled: If False, all methods are no-ops (zero overhead).
            device: CUDA device string for memory tracking.
            sync_cuda: Whether to call torch.cuda.synchronize() before each mark.
                       Required for accurate GPU timing, but adds ~0.1ms overhead per mark.
        """
        self.enabled = enabled
        self.device = device
        self.sync_cuda = sync_cuda and ('cuda' in str(device))
        self._is_cuda = 'cuda' in str(device)

        # Stage records: list of (name, elapsed_ms, mem_allocated_mb, mem_peak_mb)
        self._records = OrderedDict()
        self._last_time = None
        self._last_mem = None
        self._running = False
        self._start_time = None

        # Layer-level detail tracking
        self._layer_records = {}  # layer_idx -> OrderedDict of sub-step records
        self._current_layer = None

    def start(self, label: str = "total"):
        """Begin a profiling session. Resets all previous records."""
        if not self.enabled:
            return
        self._records = OrderedDict()
        self._layer_records = {}
        self._current_layer = None
        self._running = True
        self._start_label = label

        if self._is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        self._last_time = time.perf_counter()
        self._last_mem = self._get_mem_mb()
        self._start_time = self._last_time

    def mark(self, name: str):
        """
        Record a checkpoint. The time/memory since the last mark() or start()
        is attributed to the stage named by the PREVIOUS mark (or start).

        Call pattern:
            start("total")
            mark("stage_A")   # records time from start -> here as "stage_A"
            mark("stage_B")   # records time from stage_A -> here as "stage_B"
            stop()
        """
        if not self.enabled or not self._running:
            return

        if self.sync_cuda:
            torch.cuda.synchronize(self.device)

        now = time.perf_counter()
        mem_now = self._get_mem_mb()

        elapsed_ms = (now - self._last_time) * 1000
        mem_allocated = mem_now[0]
        mem_peak = mem_now[1]
        mem_delta = mem_allocated - self._last_mem[0]

        if self._current_layer is not None:
            # Store in layer-level records
            layer_idx = self._current_layer
            if layer_idx not in self._layer_records:
                self._layer_records[layer_idx] = OrderedDict()
            self._layer_records[layer_idx][name] = {
                'time_ms': elapsed_ms,
                'mem_alloc_mb': mem_allocated,
                'mem_peak_mb': mem_peak,
                'mem_delta_mb': mem_delta,
            }
        else:
            # Store in top-level records
            self._records[name] = {
                'time_ms': elapsed_ms,
                'mem_alloc_mb': mem_allocated,
                'mem_peak_mb': mem_peak,
                'mem_delta_mb': mem_delta,
            }

        self._last_time = now
        self._last_mem = mem_now

    def enter_layer(self, layer_idx: int):
        """Start tracking sub-steps within a transformer layer."""
        if not self.enabled:
            return
        self._current_layer = layer_idx
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)
        self._last_time = time.perf_counter()
        self._last_mem = self._get_mem_mb()

    def exit_layer(self):
        """Stop tracking sub-steps within a transformer layer."""
        if not self.enabled:
            return
        self._current_layer = None

    def stop(self):
        """End the profiling session."""
        if not self.enabled or not self._running:
            return
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)
        self._running = False
        self._total_time_ms = (time.perf_counter() - self._start_time) * 1000

    def _get_mem_mb(self):
        """Return (allocated_MB, peak_MB) for the tracked device."""
        if not self._is_cuda:
            return (0.0, 0.0)
        dev = torch.device(self.device)
        allocated = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        return (allocated, peak)

    def get_records(self) -> dict:
        """Return all profiling records as a dict."""
        return {
            'stages': dict(self._records),
            'layers': {k: dict(v) for k, v in self._layer_records.items()},
            'total_time_ms': getattr(self, '_total_time_ms', 0.0),
        }

    def summary(self, top_n_layers: int = 3):
        """Print a formatted summary table of profiling results."""
        if not self.enabled:
            print("[FEATProfiler] Profiling is disabled.")
            return

        total = getattr(self, '_total_time_ms', 0.0)

        print(f"\n{'='*90}")
        print(f"  FEAT PROFILING SUMMARY  (Total: {total:.1f}ms)")
        print(f"{'='*90}")

        # Top-level stages
        print(f"\n{'Stage':<35} | {'Time(ms)':>10} | {'%Total':>7} | {'MemAlloc(MB)':>13} | {'MemDelta(MB)':>13}")
        print(f"{'-'*35}-+-{'-'*10}-+-{'-'*7}-+-{'-'*13}-+-{'-'*13}")

        stage_total = 0
        for name, rec in self._records.items():
            pct = (rec['time_ms'] / total * 100) if total > 0 else 0
            stage_total += rec['time_ms']
            print(f"  {name:<33} | {rec['time_ms']:>10.2f} | {pct:>6.1f}% | {rec['mem_alloc_mb']:>13.1f} | {rec['mem_delta_mb']:>+13.1f}")

        # Transformer layers aggregate
        if self._layer_records:
            layer_times = {}
            for layer_idx, steps in self._layer_records.items():
                layer_time = sum(s['time_ms'] for s in steps.values())
                layer_times[layer_idx] = layer_time
                stage_total += layer_time

            transformer_total = sum(layer_times.values())
            pct = (transformer_total / total * 100) if total > 0 else 0
            print(f"  {'[Transformer Layers Total]':<33} | {transformer_total:>10.2f} | {pct:>6.1f}% | {'':>13} | {'':>13}")

            # Per-layer breakdown
            print(f"\n{'='*90}")
            print(f"  TRANSFORMER LAYER BREAKDOWN  ({len(self._layer_records)} layers)")
            print(f"{'='*90}")

            # Aggregate sub-step times across layers
            substep_totals = {}
            for layer_idx, steps in self._layer_records.items():
                for step_name, rec in steps.items():
                    if step_name not in substep_totals:
                        substep_totals[step_name] = {'time_ms': 0, 'count': 0, 'mem_delta_sum': 0}
                    substep_totals[step_name]['time_ms'] += rec['time_ms']
                    substep_totals[step_name]['count'] += 1
                    substep_totals[step_name]['mem_delta_sum'] += rec['mem_delta_mb']

            print(f"\n  Sub-step Aggregation (across all layers):")
            print(f"  {'SubStep':<30} | {'TotalTime(ms)':>13} | {'%Transformer':>12} | {'AvgTime(ms)':>12} | {'MemDelta(MB)':>13}")
            print(f"  {'-'*30}-+-{'-'*13}-+-{'-'*12}-+-{'-'*12}-+-{'-'*13}")

            for step_name, agg in sorted(substep_totals.items(), key=lambda x: -x[1]['time_ms']):
                pct = (agg['time_ms'] / transformer_total * 100) if transformer_total > 0 else 0
                avg = agg['time_ms'] / agg['count'] if agg['count'] > 0 else 0
                print(f"  {step_name:<30} | {agg['time_ms']:>13.2f} | {pct:>11.1f}% | {avg:>12.2f} | {agg['mem_delta_sum']:>+13.1f}")

            # Show top-N slowest layers
            print(f"\n  Top-{top_n_layers} Slowest Layers:")
            sorted_layers = sorted(layer_times.items(), key=lambda x: -x[1])
            for layer_idx, t in sorted_layers[:top_n_layers]:
                pct = (t / transformer_total * 100) if transformer_total > 0 else 0
                steps = self._layer_records[layer_idx]
                detail = " | ".join(f"{k}:{v['time_ms']:.1f}ms" for k, v in steps.items())
                print(f"    Layer {layer_idx:>2}: {t:>8.2f}ms ({pct:>5.1f}%)  [{detail}]")

            # Show per-layer table
            print(f"\n  Per-Layer Time Distribution:")
            print(f"  {'Layer':>6} | {'Total(ms)':>10} | ", end="")
            # Use first layer's step names as headers
            first_layer_steps = list(next(iter(self._layer_records.values())).keys())
            for step_name in first_layer_steps:
                print(f"{step_name:>12} | ", end="")
            print()
            print(f"  {'-'*6}-+-{'-'*10}-+-", end="")
            for _ in first_layer_steps:
                print(f"{'-'*12}-+-", end="")
            print()

            for layer_idx in sorted(self._layer_records.keys()):
                steps = self._layer_records[layer_idx]
                layer_total = sum(s['time_ms'] for s in steps.values())
                print(f"  {layer_idx:>6} | {layer_total:>10.2f} | ", end="")
                for step_name in first_layer_steps:
                    if step_name in steps:
                        print(f"{steps[step_name]['time_ms']:>12.2f} | ", end="")
                    else:
                        print(f"{'N/A':>12} | ", end="")
                print()

        # Memory summary
        if self._is_cuda:
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            current_mem = torch.cuda.memory_allocated(self.device) / (1024**2)
            print(f"\n{'='*90}")
            print(f"  MEMORY SUMMARY")
            print(f"{'='*90}")
            print(f"  Current GPU Allocated: {current_mem:.1f} MB")
            print(f"  Peak GPU Allocated:    {peak_mem:.1f} MB")

        print(f"\n{'='*90}\n")


# Global singleton for easy access from model internals
_global_profiler = FEATProfiler(enabled=False)

def get_profiler() -> FEATProfiler:
    """Get the global profiler instance."""
    return _global_profiler

def set_profiler(profiler: FEATProfiler):
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler

def enable_profiling(device: str = 'cuda') -> FEATProfiler:
    """Enable profiling globally and return the profiler."""
    global _global_profiler
    _global_profiler = FEATProfiler(enabled=True, device=device)
    return _global_profiler
