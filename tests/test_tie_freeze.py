"""Tests for the tie freezing in pciSeq.src.core.utils.tie_freeze.

Two things are being pinned down here.

The detector: a cell that keeps going back to a class it just held gets its
class probabilities frozen, a cell that moves once and stays does not. If the
second one ever starts firing we would be freezing cells that were still
learning, which would change the answer rather than just stopping the run.

The rebuild: check_cell does not read classProb to draw its charts, it
recomputes the posterior from the gene log-likelihood, the prior and the mrf.
For an ordinary cell the live arrays still rebuild the final classProb because
cell_to_cellType is the last thing to touch them. A pinned cell's row was
frozen partway through the run, so the freezer keeps a copy of the ingredients
from that iteration and check_cell reads those instead. The test perturbs the
live arrays after the freeze and checks the rebuild still lands on the pinned
row, because that is exactly what twenty more iterations do to them.
"""

import logging
import types

import numpy as np
import pytest
from scipy.special import softmax

from pciSeq.src.core.utils.ops_utils import cell_ingredients, gene_loglikelihood_cell
from pciSeq.src.core.utils.tie_freeze import TieFreezer, TIE_RETURNS, TIE_START


def _stub_vb(nC, nK, nG=4):
    """Just enough of a VarBayes for the freezer to read and log."""
    cells = types.SimpleNamespace(
        geneCount=np.ones((nC, nG), dtype=np.float32),
        theta_bar=np.ones((nC, nK), dtype=np.float32),
        mrf=np.zeros((nC, nK), dtype=np.float32),
        effective_beta=None,
        class_names=np.array([f"Type_{k}" for k in range(nK - 1)] + ["Zero"]),
    )
    return types.SimpleNamespace(
        cells=cells,
        genes=types.SimpleNamespace(eta_bar=np.ones(nG, dtype=np.float32)),
        cellTypes=types.SimpleNamespace(log_prior=np.zeros(nK, dtype=np.float32)),
        # no renumbering, so the segmentation label is the internal index
        config={'label_map': None},
    )


def _probs_for(labels, nK):
    """Class probabilities whose argmax is `labels`, one row per cell."""
    p = np.full((len(labels), nK), 0.1, dtype=np.float32)
    p[np.arange(len(labels)), labels] = 0.9
    return p / p.sum(axis=1, keepdims=True)


def _drive(freezer, vb, label_seq, nK):
    """Feed the freezer one iteration per row of label_seq, past TIE_START."""
    for j, labels in enumerate(label_seq):
        freezer.freeze(_probs_for(labels, nK), TIE_START + 1 + j, vb)


class TestDetector:
    """Which cells get pinned and which are left alone."""

    def test_bouncing_cell_gets_pinned(self):
        # cell 1 goes 0, 1, 0, 1: it keeps returning to a class it just held
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        _drive(freezer, _stub_vb(nC, nK), [[0, 0], [0, 1], [0, 0], [0, 1]], nK)

        assert freezer.frozen[1], "a cell that keeps going back should be pinned"
        assert freezer.snapshot(1) is not None, "a pinned cell needs its ingredients kept"

    def test_settling_cell_is_left_alone(self):
        # cell 1 moves 0 -> 1 once and stays there, which is ordinary learning
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        _drive(freezer, _stub_vb(nC, nK), [[0, 0], [0, 1], [0, 1], [0, 1]], nK)

        assert not freezer.frozen[1], "a cell that moves once and stays is not a tie"
        assert freezer.snapshot(1) is None

    def test_nothing_is_pinned_before_the_start_iteration(self):
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        vb = _stub_vb(nC, nK)
        for it, labels in enumerate([[0, 0], [0, 1], [0, 0], [0, 1]]):
            freezer.freeze(_probs_for(labels, nK), it, vb)

        assert not freezer.frozen.any(), "the early iterations are still learning"

    def test_background_is_never_pinned(self):
        # cell 0 is the background, it has no class of its own to pin
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        _drive(freezer, _stub_vb(nC, nK), [[0, 0], [1, 0], [0, 0], [1, 0]], nK)

        assert not freezer.frozen[0]

    def test_a_pinned_cell_keeps_its_row(self):
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        vb = _stub_vb(nC, nK)
        _drive(freezer, vb, [[0, 0], [0, 1], [0, 0], [0, 1]], nK)
        pinned = freezer.frozen_prob[1].copy()

        # whatever the update says from now on, the pinned row is written back
        out = freezer.freeze(_probs_for([0, 2], nK), TIE_START + 20, vb)
        assert np.allclose(out[1], pinned)
        assert out[1].argmax() != 2, "the pinned cell should ignore the new answer"

    def test_takes_more_than_one_return_to_pin(self):
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        _drive(freezer, _stub_vb(nC, nK), [[0, 0], [0, 1], [0, 0]], nK)

        assert not freezer.frozen[1]
        assert freezer.returns[1] == TIE_RETURNS - 1

    def test_log_reports_the_segmentation_label(self, caplog):
        """Internal indices mean nothing to anyone reading the log, the cell
        numbers in the viewer and in cellData are the segmentation ones. Both
        get printed so a pinned cell can actually be looked up."""
        nC, nK = 2, 3
        freezer = TieFreezer(nC, nK)
        vb = _stub_vb(nC, nK)
        # label_map goes segmentation -> internal, so internal 1 is cell 4242
        vb.config = {'label_map': {4242: 1}}

        with caplog.at_level(logging.INFO, logger='pciSeq.src.core.utils.tie_freeze'):
            _drive(freezer, vb, [[0, 0], [0, 1], [0, 0], [0, 1]], nK)

        line = [r.getMessage() for r in caplog.records if 'kept swapping' in r.getMessage()][0]
        assert 'cell 4242' in line, "the segmentation label should lead"
        assert 'internal 1' in line, "the internal index is still worth keeping"

    def test_log_names_the_two_classes_it_swapped_between(self, caplog):
        """The log used to print the top two classes of the pinned row, which
        is a different thing. A cell can have a third class sitting second on
        probability without ever having been called it, and naming that one
        makes the log say the cell was arguing with a class it never held.
        """
        nC, nK = 2, 4
        freezer = TieFreezer(nC, nK)
        vb = _stub_vb(nC, nK)

        # cell 1 swaps between Type_0 and Type_1, but on the iteration it gets
        # pinned Type_2 is the runner-up on probability
        rows = [[0, 0], [0, 1], [0, 0], [0, 1]]
        for j, labels in enumerate(rows):
            p = _probs_for(labels, nK)
            if j == len(rows) - 1:
                p[1] = np.array([0.05, 0.60, 0.30, 0.05], dtype=np.float32)
            with caplog.at_level(logging.INFO, logger='pciSeq.src.core.utils.tie_freeze'):
                freezer.freeze(p, TIE_START + 1 + j, vb)

        assert freezer.frozen[1]
        line = [r.getMessage() for r in caplog.records if 'kept swapping' in r.getMessage()]
        assert len(line) == 1, "expected exactly one cell to be pinned"
        line = line[0]

        assert 'Type_0' in line, "the class it came back from should be named"
        assert 'Type_1' in line, "the class it landed on should be named"
        assert 'Type_2' not in line, \
            "the runner-up on probability was never in the fight, it must not be named"
        assert 'pinned as Type_1' in line


class TestRebuild:
    """A pinned cell's charts have to add up to the class we report."""

    @staticmethod
    def _rebuild(vb, c, ing):
        """The posterior, recomputed the way check_cell does it."""
        contr = gene_loglikelihood_cell(vb, c, ing)
        return softmax(contr.sum(axis=0) + ing['log_prior'] + ing['mrf'])

    @staticmethod
    def _pin(vb, c):
        """Pin cell c on its real current row, by making the detector think it
        has been bouncing and then running an ordinary cell type update."""
        freezer = vb.tie_freezer
        current = vb.cells.classProb[c].argmax()
        freezer.labels[-1, c] = (current + 1) % vb.nK   # it "moved" this iteration
        freezer.labels[0, c] = current                  # back to where it was
        freezer.returns[c] = TIE_RETURNS - 1
        vb.iter_num = TIE_START + 10
        vb.cell_to_cellType()
        assert freezer.frozen[c], "the cell should have been pinned"

    def test_ordinary_cell_rebuilds_from_the_live_arrays(self, minimal_varbayes):
        # this is what makes the charts agree today, before any freezing
        vb = minimal_varbayes
        vb.initialise_state()
        vb.geneCount_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()

        c = 3
        assert vb.tie_freezer.snapshot(c) is None
        rebuilt = self._rebuild(vb, c, cell_ingredients(vb, c))
        assert np.allclose(rebuilt, vb.cells.classProb[c], atol=1e-5)

    def test_pinned_cell_rebuilds_after_the_live_arrays_move_on(self, minimal_varbayes):
        vb = minimal_varbayes
        vb.initialise_state()
        vb.geneCount_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()

        c = 3
        self._pin(vb, c)
        pinned = vb.cells.classProb[c].copy()

        # stand in for the rest of the run: the shared arrays and the cell's own
        # reads all drift after the freeze
        vb.genes._eta_bar = vb.genes.eta_bar * 1.4
        vb.cellTypes._initial_weights = vb.cellTypes._initial_weights * np.linspace(0.5, 1.5, vb.nK)
        vb.cells.theta_bar[c] = vb.cells.theta_bar[c] * 0.7
        vb.cells.geneCount[c] = vb.cells.geneCount[c] + 3.0
        vb.cells.mrf[c] = vb.cells.mrf[c] + 2.0

        rebuilt = self._rebuild(vb, c, cell_ingredients(vb, c))
        assert np.allclose(rebuilt, pinned, atol=1e-5), \
            "a pinned cell must rebuild from its stored copy, not the live arrays"

    def test_the_stored_copy_is_what_makes_the_difference(self, minimal_varbayes):
        # same setup, but rebuilding from the live arrays instead. If this ever
        # matches too then the perturbation above is not testing anything.
        vb = minimal_varbayes
        vb.initialise_state()
        vb.geneCount_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()

        c = 3
        self._pin(vb, c)
        pinned = vb.cells.classProb[c].copy()

        vb.genes._eta_bar = vb.genes.eta_bar * 1.4
        vb.cellTypes._initial_weights = vb.cellTypes._initial_weights * np.linspace(0.5, 1.5, vb.nK)
        vb.cells.mrf[c] = vb.cells.mrf[c] + 2.0

        live = dict(cell_ingredients(vb, c))
        live.update(eta_bar=vb.genes.eta_bar, log_prior=vb.cellTypes.log_prior,
                    theta_bar=vb.cells.theta_bar[c], geneCount=vb.cells.geneCount[c],
                    mrf=vb.cells.mrf[c])
        assert not np.allclose(self._rebuild(vb, c, live), pinned, atol=1e-5)

    def test_snapshot_does_not_alias_the_live_arrays(self, minimal_varbayes):
        vb = minimal_varbayes
        vb.initialise_state()
        vb.geneCount_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()

        c = 3
        self._pin(vb, c)
        before = vb.tie_freezer.snapshot(c)['eta_bar'].copy()
        vb.genes._eta_bar = vb.genes.eta_bar * 2.0

        assert np.allclose(vb.tie_freezer.snapshot(c)['eta_bar'], before), \
            "the copy must not be a view onto the live array"


class TestUnchangedBehaviour:
    """The freezer must not touch anything until it fires."""

    def test_probabilities_still_sum_to_one(self, minimal_varbayes):
        vb = minimal_varbayes
        vb.initialise_state()
        vb.geneCount_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()

        assert np.allclose(vb.cells.classProb.sum(axis=1), 1.0)
        assert not vb.tie_freezer.frozen.any(), "nothing should be pinned this early"
