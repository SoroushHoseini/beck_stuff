import argparse
import logging
import time

from base.qubit_logic import MatrixState


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and display a qubit MatrixState and its computation time"
    )
    parser.add_argument("size", type=int, help="number of spins (qubits)")
    parser.add_argument("left_sz", type=int, help="power for sz on the left subsystem")
    parser.add_argument("right_sz", type=int, help="power for sz on the right subsystem")
    parser.add_argument(
        "--partial-transpose",
        "-p",
        type=int,
        default=None,
        help="amount of partial transpose (must be <= size)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting computation for MatrixState(size={args.size}, left_sz={args.left_sz}, right_sz={args.right_sz})"
    )
    start_time = time.perf_counter()
    matrix_state = MatrixState(args.size, args.left_sz, args.right_sz)
    duration = time.perf_counter() - start_time

    if args.partial_transpose is not None:
        logger.info(f"Applying partial transpose of {args.partial_transpose}")
        matrix_state.partial_transpose(args.partial_transpose)

    n = args.size
    dim = 2**n

    all_values = list(matrix_state.matrix.values()) + [0]
    max_coeff_width = max(len(str(int(x))) for x in all_values)
    max_index_width = len(str(dim - 1))
    cell_width = max(max_coeff_width, max_index_width) + 1

    header = " " * (max_index_width + 3)
    header += "".join(f"{col:>{cell_width}}" for col in range(dim))
    print("\nMatrix coefficients (row = left index, col = right index):\n")
    print(header)
    print(" " * (max_index_width + 3) + "-" * (cell_width * dim))

    for row in range(dim):
        row_label = f"{row:>{max_index_width}} |"
        row_entries = []
        for col in range(dim):
            coeff = matrix_state.matrix.get((row, col), 0)
            row_entries.append(f"{coeff:>{cell_width}}")
        print(f"{row_label}{''.join(row_entries)}")
    print()

    eigs = matrix_state.eigenvalues
    if eigs is not None:
        print(f"Eigenvalues ({len(eigs)}):")
        line = []
        for i, eig in enumerate(eigs):
            line.append(f"{eig:.6g}")
            if (i + 1) % 8 == 0:
                print("  " + "  ".join(line))
                line = []
        if line:
            print("  " + "  ".join(line))
    else:
        print("Eigenvalues could not be computed.")

    if matrix_state.negativity is not None:
        print(f"\nNegativity: {matrix_state.negativity:.6g}")
    else:
        print("\nNegativity could not be computed.")

    logger.info(f"Computed matrix in {duration:.6f} seconds")


if __name__ == "__main__":
    main()
