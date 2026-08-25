"""Pins cells that keep flip-flopping between two equally good classes.

The problem this solves is NOT the model failing to converge. The ELBO
plateaus early and stays there: measured on the reference run it moves by about
one part in 50 million per iteration after iteration 150, and a cell changing
its class outright moves it by less than that. The chain has converged. What
has not settled is which of two labels a handful of cells report, and since the
two labels are worth the same to the objective, the model has no reason to
prefer either and no amount of further fitting will produce one.

The run keeps going anyway because the stopping rule watches the largest
spot-to-cell probability change in the whole dataset. A few dozen cells holding
a handful of reads each keep throwing their spots back and forth, so that
number keeps being dragged back above the tolerance, and 26k settled cells wait
on them.

See the worked example in TieFreezer for how a cell gets into that state.
"""
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

    What the ping-pong looks like
    -----------------------------
    Everything below is measured on a run with the mrf switched off, so none
    of it is a neighbourhood effect.

    A single spot, one iteration apart::

        iter 53   cell 9667  -> background    p 0.3721 -> 0.3693
        iter 54   background -> cell 9667     p 0.3693 -> 0.3771

    Given to the cell, given back, given again. The probability hardly moves,
    0.37 either way, because the spot really could belong to either owner. It
    is a coin standing on its edge and the argmax has to report a face. 92
    spots do this in that run.

    On its own that costs nothing, since the probability is not moving. What
    holds a run up is a whole cell changing its mind at once::

        iter   delta    what happened
         163   0.0248   cell 9027 is Zero, as it has been since iteration 20
         164   0.0442   it flips to 329 ABC NN
         165   0.0952   spot 3066431: background -> cell 9027  p 0.397 -> 0.443
         166   0.2434   spot 3066430: background -> cell 9027  p 0.370 -> 0.540
         167   0.3966   spot 3066445: background -> cell 9027  p 0.408 -> 0.625
         168   0.1351   three more spots follow
         172   0.0114   quiet again

    Zero has no gene preference, so a cell calling itself Zero competes for
    nothing. The moment it calls itself a real class it starts competing, and
    it takes spots off the background and off its neighbours, each by a wide
    margin. The stopping rule watches the single largest spot move anywhere in
    3.1 million spots, so this one cell set that number for the whole dataset
    for nine iterations, at twenty times the tolerance on the worst of them.

    Then it goes quiet, and twenty or so iterations later another cell does the
    same thing. Over 250 iterations only 52 came in under the tolerance: the
    run reaches the line, bounces off it, and comes back. That is the plateau.

    Why pinning is not cheating
    ---------------------------
    We are not choosing a winner the data could have decided; we are recording
    that there is no winner to decide. The two states differ by less than one
    part in ten million of the objective, so no further fitting would separate
    them, and the cell would keep swapping for as long as you let it run. What
    the pin does is take one of the two answers the model actually produced and
    stop asking again.

    Three things keep it above board. Nothing is pinned before TIE_START, so
    cells still learning are untouched, and a run is bit-for-bit identical to
    an unpinned one until the first pin fires. What gets pinned is a real
    iteration of the model, never an average, so it has real ingredients behind
    it and check_cell can still explain it. And every pinned cell is logged
    with its reads, the two classes it was swapping between and the
    probabilities, so the decision is visible rather than silent.

    How they are detected
    ---------------------
    A cell that is still learning moves to a new class and stays, a tie cell
    goes back to where it just was (e.g. Oligo, Zero, Oligo, Zero, ...). So
    once a cell has come back twice to a class it held in the last TIE_HISTORY
    iterations, we take it as a tie and pin it. Two real classes arguing counts
    the same as a swap with Zero: either way the cell will never settle, and
    either way it would keep the run going for ever.

    Note the detector only counts a return on an iteration where the cell
    actually moved, so a cell that swaps every iteration is pinned as soon as
    the rule allows, and a slower one takes proportionally longer.

    Why a real iteration and not an average
    ---------------------------------------
    check_cell does not read classProb to draw its charts, it recomputes the
    posterior from the ingredients that went into it, and the two agree today
    only because cell_to_cellType is the last thing to touch those ingredients
    before the run exits. An average of five iterations is not something the
    softmax ever produced, so nothing would ever add up to it and the charts
    could not be made to agree. A single real iteration always has ingredients
    behind it.

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
        # internal index -> segmentation label, for the log lines only. Built
        # on first use because the config is not around at construction time.
        self._seg_labels = None

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
                # self.labels[-1] is still last iteration's, the roll is below
                self._log(vb, idx, pCellClass, labels, self.labels[-1], it)

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

    def _seg_label(self, vb, c: int):
        """The segmentation label of an internal index, so the log says the
        same cell number you see in the viewer and in cellData.

        config['label_map'] goes segmentation -> internal, so it has to be
        turned round. It is None when the labels were already sequential and
        nothing was renumbered, and then the two numbers are the same anyway.
        """
        if self._seg_labels is None:
            lm = vb.config.get('label_map')
            self._seg_labels = {v: k for k, v in lm.items()} if lm else {}
        return self._seg_labels.get(int(c), int(c))

    def _log(self, vb, frozen_cells, pCellClass, labels, prev_labels, it) -> None:
        """One log line per freshly pinned cell: its reads, the two classes it
        was swapping between, and which one it ended up pinned on.

        The two classes are the one the cell holds now and the one it held last
        iteration, which are the two sides of the swap that set the pin off.
        That is not the same as the top two of the pinned row: some third class
        can be sitting second on probability without the cell ever being called
        it, so reading the row would name a class that was never in the fight.
        """
        # everything is turned into a plain str or int first. numpy scalars go
        # through %d and %.2f fine on their own, but not under every logging
        # handler, and a broken log line should not take a run down with it.
        cells = vb.cells
        reads = cells.geneCount[frozen_cells].sum(axis=1)
        for j, c in enumerate(frozen_cells):
            here, there = int(labels[c]), int(prev_labels[c])
            logger.info(
                "    cell %s (internal %s, %s reads) kept swapping between %s and %s, "
                "pinned as %s (p=%s)",
                self._seg_label(vb, c), int(c), "%.2f" % float(reads[j]),
                str(cells.class_names[here]), str(cells.class_names[there]),
                str(cells.class_names[here]), "%.2f" % float(pCellClass[c, here]))
        logger.info("Iteration %s: pinned %s cell(s), %s pinned in total",
                    int(it), len(frozen_cells), int(self.frozen.sum()))
