"""Distributed execution helpers for Optuna / heavy data.

This module provides a small, optional scaffold to integrate Dask or Ray
for distributed data processing and parallel Optuna execution.

The functions are intentionally lightweight so importing Octolearn
remains dependency-free; users can opt-in by installing the
`distributed` extras (dask, ray).
"""
from typing import Optional

def detect_backend() -> Optional[str]:
    """Detect available distributed backend. Returns 'dask', 'ray', or None."""
    try:
        import dask  # type: ignore
        return "dask"
    except Exception:
        pass
    try:
        import ray  # type: ignore
        return "ray"
    except Exception:
        return None


def setup_distributed_backend(backend: str = "auto", **kwargs) -> str:
    """Attempt to set up the requested backend.

    Parameters
    - backend: 'auto'|'dask'|'ray'
    - kwargs: passed to backend initializer if applicable

    Returns the selected backend string.
    """
    if backend == "auto":
        found = detect_backend()
        if found is None:
            raise RuntimeError("No distributed backend (dask or ray) is installed.")
        backend = found

    if backend == "dask":
        try:
            from dask.distributed import Client  # type: ignore
            client = Client(**kwargs)
            return "dask"
        except Exception as e:
            raise RuntimeError(f"Failed to start Dask client: {e}")

    if backend == "ray":
        try:
            import ray  # type: ignore
            if not ray.is_initialized():
                ray.init(**kwargs)
            return "ray"
        except Exception as e:
            raise RuntimeError(f"Failed to start Ray: {e}")

    raise ValueError("Unsupported backend: %s" % backend)


def run_optuna_with_backend(study, objective_fn, backend: str = "auto", **backend_opts):
    """Helper to run an Optuna study using a distributed backend when available.

    This is a lightweight shim. For full distributed Optuna integration
    users should use Optuna's Dask or Ray integration directly.
    """
    selected = setup_distributed_backend(backend, **backend_opts)
    # This is intentionally non-prescriptive: return the selected backend
    # and let the caller manage the study.run() with their preferred integration.
    return selected
