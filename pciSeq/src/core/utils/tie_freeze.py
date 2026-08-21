"""Freezes cells that keep flip-flopping between the same classes."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Settings for the tie freezing, see TieFreezer. They are not in config.py
# because they are not model parameters, they only decide when a cell is
# called a tie.
TIE_START = 50    # leave the early iterations alone, the run is still learning
TIE_HISTORY = 5   # how many past iterations of labels to keep
TIE_RETURNS = 2   # how many times a cell must come back before we freeze it


class TieFreezer:
    """Pins the class probabilities of cells that never make up their mind.

    The ELBO flattens early, but a few nearly empty cells keep swapping class
    every iteration, so the run never passes its convergence check. Each swap
    drags the spots around them from the cell to the background and back, and
    the stopping rule looks at the biggest spot move of the iteration, so a
    couple of dozen cells hold up the whole run.

    A cell that is still learning moves to a new class and stays, a tie cell
    goes back to where it just was (e.g. Oligo, Zero, Oligo, Zero, ...). So
    once a cell has come back twice to a class it held in the last TIE_HISTORY
    iterations, we take it as a tie and pin it. Two real classes arguing counts
    the same as a swap with Zero: either way the cell will never settle, and
    either way it would keep the run going for ever.

    What gets pinned is the state the cell is in at that moment, not an average
    of its recent ones. That matters for the diagnostics. check_cell does not
    read classProb to draw its charts, it recomputes the posterior from the
    ingredients that went into it, and the two agree today only because
    cell_to_cellType is the last thing to touch those ingredients before the
    run exits. An average of five iterations is not something the softmax ever
    produced, so nothing would ever add up to it and the charts could not be
    made to agree. A single real iteration always has ingredients behind it.

    Those ingredients are what snapshots holds. They have to be copied because
    the live ones carry on moving after the freeze: eta_bar and log_prior are
    shared with the other 26k cells, which are still learning, and the cell's
    own reads still drift as its neighbours settle. Twenty iterations later
    they would no longer rebuild the pinned row. The copy does not move, so
    check_cell recomputes from it exactly as it always has and lands back on
    the pinned row.

    Call freeze() once per iteration, at the end of the cell type update:

        freezer = TieFreezer(nC, nK)
        pCellClass = freezer.freeze(pCellClass, iter_num, vb)
    """

    def __init__(self, nC: int, nK: int) -> None:
        """The labels start at -1 so that a cell cannot match against a slot
        that has not been filled in yet."""
        self.labels = np.full((TIE_HISTORY, nC), -1, dtype=np.int32)
        self.returns = np.zeros(nC, dtype=np.int16)
        self.frozen = np.zeros(nC, dtype=bool)
        self.frozen_prob = np.zeros((nC, nK), dtype=np.float32)
        # everything cell_to_cellType needs to rebuild a pinned cell's row,
        # frozen at the iteration it was pinned. One entry per pinned cell,
        # see _snapshot for what is in it. A dict rather than full arrays
        # because this is a couple of dozen cells out of nC.
        self.snapshots = {}

    def freeze(self, pCellClass, it, vb) -> np.ndarray:
        """Apply the tie freezing to one iteration of class probabilities.

        Args:
            pCellClass: (nC, nK) class probabilities from the cell type update.
            it: iteration number; nothing is checked before TIE_START.
            vb: the VarBayes object, read for the log lines and for the
                ingredients copied when a cell is pinned. The cell type update
                has just refreshed all of them for this iteration.

        Returns:
            pCellClass, with previously pinned rows written back and any newly
            tied cells pinned to the state they are in now.
        """
        # a pinned cell keeps its row, whatever the update above said
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

            newly = came_back & (self.returns >= TIE_RETURNS)
            newly[0] = False  # cell 0 is the background, it has no class to pin
            if newly.any():
                idx = np.flatnonzero(newly)
                self.frozen[idx] = True
                self.frozen_prob[idx] = pCellClass[idx]
                for c in idx:
                    self.snapshots[int(c)] = self._snapshot(vb, int(c), it)
                self._log(vb.cells, idx, pCellClass, it)

        # keep the history rolling from the very first iteration, so it is
        # already filled by the time we start checking
        self.labels = np.roll(self.labels, -1, axis=0)
        self.labels[-1] = labels
        return pCellClass

    @staticmethod
    def _snapshot(vb, c: int, it: int) -> dict:
        """Copy of everything needed to rebuild cell c's pinned row.

        These are the inputs to cell_to_cellType, not the answer it gave, so
        check_cell and the viewer can run their usual computation on them.
        scaled_exp is not here because it is area_factor * mean_expression and
        does not change during the run, and rSpot / SpotReg are config.

        The mrf row is the capped one, the term the update actually used,
        so a pinned cell's chart shows the mrf that decided it.
        """
        beta = vb.cells.effective_beta
        return {
            'iteration': it,
            'eta_bar': np.asarray(vb.genes.eta_bar, dtype=np.float32).copy(),
            'log_prior': np.asarray(vb.cellTypes.log_prior, dtype=np.float32).copy(),
            'theta_bar': np.asarray(vb.cells.theta_bar[c], dtype=np.float32).copy(),
            'geneCount': np.asarray(vb.cells.geneCount[c], dtype=np.float32).copy(),
            'mrf': np.asarray(vb.cells.mrf[c], dtype=np.float32).copy(),
            'effective_beta': (None if beta is None else
                               np.asarray(beta[c], dtype=np.float32).copy()),
        }

    def snapshot(self, c: int):
        """The stored ingredients for cell c, or None if it was never pinned."""
        return self.snapshots.get(int(c))

    def _log(self, cells, frozen_cells, pCellClass, it) -> None:
        """One log line per freshly pinned cell: its reads and where it landed."""
        # everything is turned into a plain str or int first. numpy scalars go
        # through %d and %.2f fine on their own, but not under every logging
        # handler, and a broken log line should not take a run down with it.
        reads = cells.geneCount[frozen_cells].sum(axis=1)
        for j, c in enumerate(frozen_cells):
            row = pCellClass[c]
            first, second = np.argsort(-row)[:2]
            logger.info(
                "    cell %s (internal id, %s reads) kept swapping between %s and %s, "
                "pinned as %s (p=%s)",
                int(c), "%.2f" % float(reads[j]),
                str(cells.class_names[first]), str(cells.class_names[second]),
                str(cells.class_names[first]), "%.2f" % float(row[first]))
        logger.info("Iteration %s: pinned %s cell(s), %s pinned in total",
                    int(it), len(frozen_cells), int(self.frozen.sum()))
