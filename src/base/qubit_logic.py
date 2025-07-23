import logging
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


class SpinState:
    """
    Represents an n-spin system superposition of bit states with integer coefficients.
    Each basis state is encoded as an integer from 0 to 2^n - 1.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be a positive integer")
        self.size: int = size
        self.state: Counter[int] = Counter({0: 1})
        logger.info(f"Initialized SpinState(size={self.size}) with state={{0: 1}}")

    def sz(self, power: int) -> None:
        if power < 1:
            raise ValueError("power must be a positive integer")
        for p in range(1, power + 1):
            new_state: Counter[int] = Counter()
            for basis, coeff in self.state.items():
                for bit in range(self.size):
                    flipped = basis ^ (1 << bit)
                    new_state[flipped] += coeff
            self.state = new_state
            logger.info(f"After sz({p}), state has {len(self.state)} terms")


class MatrixState:
    """
    Represents a tensor-product matrix of two SpinState systems of equal size.
    Internally stored as a sparse mapping from (row_index, col_index) to coefficient.

    Additionally computes and stores:
      - Dense numpy array representation of the matrix.
      - Eigenvalues of the matrix.
      - The matrix normalized by its trace (as a numpy array).
      - Negativity: sum of negative eigenvalues of the normalized matrix.
    """

    def __init__(self, size: int, left_sz_power: int, right_sz_power: int) -> None:
        if size < 1:
            raise ValueError("size must be a positive integer")
        self.size: int = size
        logger.info(
            f"Creating MatrixState(size={size}, left_sz={left_sz_power}, right_sz={right_sz_power})"
        )

        self.left = SpinState(size)
        self.right = SpinState(size)

        logger.info("Applying sz to left subsystem...")
        self.left.sz(left_sz_power)
        logger.info("Applying sz to right subsystem...")
        self.right.sz(right_sz_power)

        self.matrix: dict[tuple[int, int], int] = {}
        for i, ci in self.left.state.items():
            for j, cj in self.right.state.items():
                key = (i, j)
                self.matrix[key] = self.matrix.get(key, 0) + ci * cj
        logger.info(f"Tensor-product matrix constructed with {len(self.matrix)} nonzero entries")

        self.eigenvalues: list[float] | None = None
        self.normalized_matrix: np.ndarray | None = None
        self.negativity: float | None = None
        self._update_analysis()

    def _update_analysis(self) -> None:
        """
        Updates all derived data:
          - eigenvalues (real part, sorted)
          - normalized matrix (dense, by trace)
          - negativity (sum of negative eigenvalues of normalized matrix)
        """
        dim = 2**self.size
        arr = np.zeros((dim, dim), dtype=float)
        for (i, j), coeff in self.matrix.items():
            arr[i, j] = coeff
        try:
            eigs = np.linalg.eigvals(arr)
            self.eigenvalues = sorted(eigs.real.tolist())
            logger.info(f"Eigenvalues updated ({len(self.eigenvalues)} total)")
        except Exception as exc:
            logger.error(f"Eigenvalue computation failed: {exc}")
            self.eigenvalues = None

        tr = float(np.trace(arr))
        if abs(tr) > 1e-12:
            self.normalized_matrix = arr / tr
            try:
                norm_eigs = np.linalg.eigvals(self.normalized_matrix)
                self.negativity = float(np.sum(norm_eigs.real[norm_eigs.real < 0]))
                logger.info(f"Negativity updated: {self.negativity:.6g}")
            except Exception as exc:
                logger.error(f"Normalized eigenvalue computation failed: {exc}")
                self.negativity = None
        else:
            self.normalized_matrix = None
            self.negativity = None
            logger.warning("Matrix trace is zero; cannot normalize.")

    def partial_transpose(self, k: int) -> None:
        if k < 0 or k > self.size:
            raise ValueError("k must be between 0 and size")
        mask = (1 << k) - 1
        new_matrix: dict[tuple[int, int], int] = {}
        for (i, j), coeff in self.matrix.items():
            i_low = i & mask
            i_high = i >> k
            j_low = j & mask
            j_high = j >> k
            new_i = (i_high << k) | j_low
            new_j = (j_high << k) | i_low
            new_matrix[(new_i, new_j)] = new_matrix.get((new_i, new_j), 0) + coeff
        self.matrix = new_matrix
        logger.info(
            f"Performed partial transpose(k={k}), new matrix has {len(self.matrix)} nonzero entries"
        )
        self._update_analysis()
