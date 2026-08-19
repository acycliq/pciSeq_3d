"""Freezes cells that keep flip-flopping between Zero and a real class."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Settings for the tie freezing, see TieFreezer. They are not in config.py
# because they are not model parameters, they only decide when a cell is
# called a tie.
TIE_START = 50    # leave the early iterations alone, the run is still learning
TIE_HISTORY = 5   # how many past iterations of labels and class probs to keep
TIE_RETURNS = 2   # how many times a cell must come back before we freeze it


class TieFreezer:
    """Pins the class probabilities of cells stuck flip-flopping with Zero.

    The ELBO flattens early, but a few nearly empty cells on the Zero
    boundary keep swapping class every iteration, so the run never passes
    its convergence check. A cell that is still learning moves somewhere
    new and stays; a tie cell goes back to where it just was (e.g.
    Oligo, Zero, Oligo, Zero, ...). So once a cell has come back twice to
    a class it held in the last TIE_HISTORY iterations, and Zero is among
    them, we freeze its probabilities to the mean of those recent rows (an even
    mix of both sides of the swap) and keep that row for the rest of the run.
    Cells argued over by two real classes are left alone.

    Keeps the labels and class probs of the last TIE_HISTORY iterations;
    call freeze() once per iteration:

        freezer = TieFreezer(nC, nK)
        pCellClass = freezer.freeze(pCellClass, iter_num, cells)
    """

    def __init__(self, nC: int, nK: int) -> None:
        """The labels start at -1 so that a cell cannot match against a slot
        that has not been filled in yet."""
        self.nK = nK
        self.labels = np.full((TIE_HISTORY, nC), -1, dtype=np.int32)
        self.probs = np.zeros((TIE_HISTORY, nC, nK), dtype=np.float32)
        self.returns = np.zeros(nC, dtype=np.int16)
        self.frozen = np.zeros(nC, dtype=bool)
        self.frozen_prob = np.zeros((nC, nK), dtype=np.float32)

    def freeze(self, pCellClass, it, cells) -> np.ndarray:
        """Apply the tie freezing to one iteration of class probabilities.

        Args:
            pCellClass: (nC, nK) class probabilities from the cell type update.
            it: iteration number; nothing is checked before TIE_START.
            cells: the Cells object, only used for the log lines (geneCount
                and class_names).

        Returns:
            pCellClass, with previously frozen rows pinned back and the rows
            of any newly tied cells frozen to their recent average.
        """
        # a frozen cell keeps its pinned row, whatever the update above said
        if self.frozen.any():
            pCellClass[self.frozen] = self.frozen_prob[self.frozen]

        labels = pCellClass.argmax(axis=1)
        # iter_num is None until main_loop starts and the tests call the cell
        # type update straight after initialise_state, so None can reach here
        it = it if it is not None else 0

        if it > TIE_START:
            # a cell "came back" if the class it just moved to is one it held
            # in the last few iterations
            moved = (labels != self.labels[-1]) & ~self.frozen
            came_back = moved & (self.labels == labels).any(axis=0)
            self.returns[came_back] += 1

            # freeze the cells that came back twice while swapping with Zero
            # (the last column); two real classes arguing is left alone
            zero = self.nK - 1
            to_freeze = (came_back
                         & (self.returns >= TIE_RETURNS)
                         & ((self.labels == zero).any(axis=0) | (labels == zero)))
            if to_freeze.any():
                mean_prob = (self.probs[:, to_freeze].sum(axis=0)
                             + pCellClass[to_freeze]) / (TIE_HISTORY + 1)
                mean_prob /= mean_prob.sum(axis=1, keepdims=True)
                pCellClass[to_freeze] = mean_prob
                self.frozen[to_freeze] = True
                self.frozen_prob[to_freeze] = mean_prob
                labels[to_freeze] = mean_prob.argmax(axis=1)
                self._log(cells, np.flatnonzero(to_freeze), mean_prob, it)

        # keep the history rolling from the very first iteration, so it is
        # already filled by the time we start checking
        self.labels = np.roll(self.labels, -1, axis=0)
        self.labels[-1] = labels
        self.probs = np.roll(self.probs, -1, axis=0)
        self.probs[-1] = pCellClass
        return pCellClass

    def _log(self, cells, frozen_cells, mean_prob, it) -> None:
        """One log line per freshly frozen cell: its reads and the tied classes."""
        reads = cells.geneCount[frozen_cells].sum(axis=1)
        for j, c in enumerate(frozen_cells):
            first, second = np.argsort(-mean_prob[j])[:2]
            logger.info(
                "    cell %d (%.2f reads) is a tie between %s (p=%.2f) and %s (p=%.2f)",
                c, reads[j],
                cells.class_names[first], mean_prob[j, first],
                cells.class_names[second], mean_prob[j, second])
        logger.info("Iteration %d: froze %d cell(s), %d frozen in total",
                    it, len(frozen_cells), int(self.frozen.sum()))
