import numpy as np

try:
    import cupy as cp
    _t = cp.ones((2, 2))
    cp.dot(_t, _t)
    GPU_AVAILABLE = True
    print("[GPU] CuPy + CUDA disponible - calculs sur GPU")
except ModuleNotFoundError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("[CPU] CuPy non installé - calculs sur CPU  →  pip install cupy-cuda12x")
except Exception as e:
    import numpy as cp
    GPU_AVAILABLE = False
    print(f"[CPU] CuPy présent mais CUDA non fonctionnel ({type(e).__name__}: {e})")


def to_gpu(arr):
    return cp.asarray(arr)


def to_cpu(arr):
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def liberer_gpu():
    """Vide les pools mémoire CuPy (GPU + mémoire hôte épinglée)."""
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
