import logging
from collections import Counter
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

class PhotonState:
    """
    Represents a two-mode photon Fock state |Bx, By> with coefficients, allowing superpositions.
    State is stored as Counter mapping (Bx, By) tuples to float coefficients.
    """

    def __init__(self, Bx: int, By: int) -> None:
        if Bx < 0 or By < 0:
            raise ValueError("Bx and By must be non-negative integers")
        self.state: Counter[tuple[int, int]] = Counter({(Bx, By): 1.0})
        logger.info(f"Initialized PhotonState with |{Bx}, {By}>")

    def Jz(self) -> None:
        """
        Applies the Jz operator: Jz = ax*ay - ay*ax to the current state.
        Updates the state in-place. Jz|Bx, By> = sqrt((Bx+1)*By)|Bx+1, By-1> - sqrt((By+1)*Bx)|Bx-1, By+1>
        """
        new_state: Counter[tuple[int, int]] = Counter()
        for (bx, by), coeff in self.state.items():
            if by > 0:
                amp1 = math.sqrt(bx + 1) * math.sqrt(by)
                state1 = (bx + 1, by - 1)
                new_state[state1] += coeff * amp1
                logger.debug(f"Jz term1: |{bx},{by}> -> {amp1:+.3f}|{state1[0]},{state1[1]}>")
            if bx > 0:
                amp2 = math.sqrt(by + 1) * math.sqrt(bx)
                state2 = (bx - 1, by + 1)
                new_state[state2] -= coeff * amp2
                logger.debug(f"Jz term2: |{bx},{by}> -> {-amp2:+.3f}|{state2[0]},{state2[1]}>")
        self.state = new_state
        logger.info(f"After Jz, state has {len(self.state)} terms: {self.state}")

    def __repr__(self) -> str:
        parts = []
        for (bx, by), c in sorted(self.state.items()):
            parts.append(f"{c:+.3f}|{bx},{by}>")
        return " + ".join(parts) if parts else "0"

