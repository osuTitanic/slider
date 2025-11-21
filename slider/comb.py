
import math
import numpy as np
import numpy.typing as npt

def comb(n: int, k: int | npt.NDArray[np.int64]) -> int | npt.NDArray[np.float64]:
    if isinstance(k, np.ndarray):
        return np.array([_math_comb(n, int(ki)) for ki in k], dtype=float)
    return _math_comb(n, k)

def _comb_fallback(n: int, k: int | npt.NDArray[np.int64]) -> int | npt.NDArray[np.float64]:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

_math_comb = getattr(math, "comb", _comb_fallback)
