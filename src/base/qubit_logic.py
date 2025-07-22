import logging
from collections import Counter
from typing import Dict, Tuple

# configure logging to output to terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class SpinState:
    """
    Represents an n-spin system superposition of bit states with integer coefficients.
    Each basis state is encoded as an integer from 0 to 2^n - 1.
    """

    def __init__(self, size: int) -> None:
        """
        Initialize a spin state of given size with all spins set to 0.
        The internal state is a Counter mapping basis-index to coefficient.

        :param size: number of spins (must be positive)
        """
        if size < 1:
            raise ValueError("size must be a positive integer")
        self.size: int = size
        self.state: Counter[int] = Counter({0: 1})
        logger.info(f"Initialized SpinState(size={self.size}) with state={{0: 1}}")

    def sz(self, power: int) -> None:
        """
        Apply the 'single-spin flip' operator power times.
        Each application flips each spin once, expanding the superposition,
        and collects like terms by summing coefficients.

        :param power: number of times to apply the flip operation (must be positive)
        """
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
    """

    def __init__(self, size: int, left_sz_power: int, right_sz_power: int) -> None:
        """
        Initialize the matrix state by creating two SpinState objects,
        applying sz on each, then building their tensor-product.

        :param size: number of spins in each subsystem (must be positive)
        :param left_sz_power: number of sz applications on the left subsystem
        :param right_sz_power: number of sz applications on the right subsystem
        """
        if size < 1:
            raise ValueError("size must be a positive integer")
        self.size: int = size
        logger.info(f"Creating MatrixState(size={size}, left_sz={left_sz_power}, right_sz={right_sz_power})")

        self.left = SpinState(size)
        self.right = SpinState(size)

        logger.info("Applying sz to left subsystem...")
        self.left.sz(left_sz_power)
        logger.info("Applying sz to right subsystem...")
        self.right.sz(right_sz_power)

        self.matrix: Dict[Tuple[int, int], int] = {}
        for i, ci in self.left.state.items():
            for j, cj in self.right.state.items():
                key = (i, j)
                self.matrix[key] = self.matrix.get(key, 0) + ci * cj
        logger.info(f"Tensor-product matrix constructed with {len(self.matrix)} nonzero entries")

    def partial_transpose(self, k: int) -> None:
        """
        Perform the partial transpose operation by swapping the last k bits
        between row and column indices in the matrix keys.

        :param k: number of least-significant bits to transpose (0 <= k <= size)
        """
        if k < 0 or k > self.size:
            raise ValueError("k must be between 0 and size")
        mask = (1 << k) - 1
        new_matrix: Dict[Tuple[int, int], int] = {}
        for (i, j), coeff in self.matrix.items():
            i_low = i & mask
            i_high = i >> k
            j_low = j & mask
            j_high = j >> k
            new_i = (i_high << k) | j_low
            new_j = (j_high << k) | i_low
            new_matrix[(new_i, new_j)] = new_matrix.get((new_i, new_j), 0) + coeff
        self.matrix = new_matrix
        logger.info(f"Performed partial transpose(k={k}), new matrix has {len(self.matrix)} nonzero entries")

