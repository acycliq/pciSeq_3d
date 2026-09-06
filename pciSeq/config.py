"""
Settings for pciSeq.

Pass any of these to pciSeq.fit() in an opts dictionary. Anything you leave out
keeps the default below.
"""


DEFAULT = {


    # Genes to leave out of cell calling, e.g. ['Aldoc', 'Id2']. Every spot of an
    # excluded gene is dropped before anything else runs.
    "exclude_genes": [],


    # Most runs settle well before this. It is a safety net so a run that will
    # not converge still terminates.
    "max_iter": 1000,


    # When to stop. After each pass the algorithm looks at how far the spot to
    # cell assignment probabilities moved, and stops once that is below this
    # number. The default is 0.02 (ie 2%) which means that the algorithm will
    # not stop until all spot -> cell assignment probabilities change by max 2%
    # between two successive loops
    "CellCallTolerance": 0.02,


    # Inefficiency is the detection efficiency of the in situ experiment relative
    # to the single cell reference, and usually the first setting worth adjusting.
    #
    # In situ sequencing sees only a fraction of the RNA that scRNA-seq reports.
    # At 0.2 a gene the reference says has 100 copies is expected to give about
    # 20 spots. Unlike everything else here this is applied straight across the
    # whole reference, every gene in every class, so it sets the overall scale
    # the model expects your counts to be on.
    #
    # Raise it and the model expects more spots per cell, so a cell needs a
    # higher count before it looks like a given class, and thin cells are more
    # readily left to the Zero class. Lower it and the same cells look better
    # populated. If your cells are being called Zero far more often than the
    # tissue suggests they should be, or conversely if almost nothing is landing
    # in Zero, this is the setting to look at first.
    #
    # 0.2 is a reasonable starting point for in situ sequencing, and worth
    # revisiting for a different chemistry or a different tissue.
    #
    # This is the first of four corrections that multiply together to give the
    # reads a cell is expected to show, each working at a finer level than the
    # one before it:
    #
    #   Inefficiency   the whole panel at once
    #   eta            one gene, across every cell     (weight rGene)
    #   theta          one cell, across every gene     (weight rTheta)
    #   gamma          one gene in one cell            (weight rSpot)
    #
    # Only Inefficiency is a number you set outright. The other three are fitted
    # from your data, and what you set is how strongly each is held to 1, meaning
    # no correction at that level.
    "Inefficiency": 0.2,


    # Some probes work better than others, so the second correction, eta, works
    # on one gene across every cell. Eta is the gene's observed read count
    # divided by the count the model expects for it. Above 1 the gene is picked
    # up better than the panel rate, below 1 worse, and at 1 the two agree and
    # nothing needs adjusting.
    #
    # The prior puts eta at 1, and rGene decides how firmly it is held there.
    # What matters is the value of rGene compared to the gene's own read count
    # across the whole section. Set it very large and eta stays at 1 for
    # every gene, so the whole panel sits at the Inefficiency rate. At 20 rGene
    # has almost no influence, because a gene picks up thousands of reads once
    # they are counted section wide, and the data determine eta.
    "rGene": 20,


    # Some cells express more transcripts than the reference predicts and others
    # fewer, so the third correction, theta, works on one cell across every gene.
    # Theta is the cell's observed read count divided by the count its candidate
    # class predicts. Above 1 the cell has more reads than the class predicts,
    # below 1 it has fewer, and at 1 the two agree and nothing needs rescaling.
    #
    # The prior puts theta at 1, and rTheta decides how firmly it is held there.
    # What matters is the value of rTheta compared to the cell's own read count.
    # Set it very large and theta stays at 1 for every cell, so the reference is
    # used as it comes. Set it to something like 2 and rTheta has almost no
    # influence, because nearly every cell has far more reads than that and the data
    # determine theta.
    #
    # It has to stay above 1. Theta is worked out as (reads + rTheta - 1) over
    # (rTheta + expected), so at rTheta of 1 or less a cell with no assigned reads
    # comes out with a theta of zero or below, which is meaningless and breaks the
    # arithmetic further on. Values at or below 1 are rejected.
    "rTheta": 25.0,


    # Gene counts vary more between cells of one class than a Poisson would allow,
    # so they are modelled as negative binomial. rSpot is the dispersion: lower
    # values mean cells of the same class differ more from each other, higher
    # values mean they look alike. 2 fits typical in situ data.
    "rSpot": 2,


    # An extra push toward a cell when the spot falls inside that cell's
    # segmented boundary, added on the log scale to the spot to cell score. 0
    # switches it off, which is the default: the distance term already prefers
    # nearby cells, and a boundary bonus makes the answer depend on exactly where
    # the segmentation drew its outlines. Passing True is accepted and quietly
    # turned into 2.
    "InsideCellBonus": 0,


    # How strongly a cell's neighbours pull it toward their own class.
    #
    # Cells of the same class tend to sit together, so a class already common
    # around a cell gets a bonus when that cell is scored. Each neighbour
    # contributes in proportion to how close it is and how sure it is of its own
    # class.
    #
    # This helps most in tissue with clear spatial organisation, cortical layers
    # or the hippocampus for instance, where a cell's neighbours really do say
    # something about what it is. It has less to add where the classes are mixed
    # together rather than grouped.
    #
    # Set it to 0 to switch the mrf off.
    "mrf_beta": 1.0,


    # MisreadDensity sets the background level. Not every read comes from a cell:
    # some RNA sits in processes too far from any soma to attribute, and some
    # reads are technical misreads. Those are modelled as a background spread
    # evenly over the whole region, and every read is offered that option
    # alongside the nearby cells. A read goes to a cell only if that cell
    # explains it better than the background would.
    #
    # A separate density is estimated for each gene, because genes differ in how
    # noisy they are, and the value here is the starting point those estimates
    # are pulled toward. rRho controls how hard they are pulled.
    #
    # The number looks uncomfortably small because it is a density, misreads per
    # unit of imaged volume, and the volumes are large. To picture it, take a
    # block of 100 x 100 pixels over 10 planes with a voxel size of
    # [0.28, 0.28, 0.7]. The z step is 2.5 times the pixel, so that block counts
    # as 100 x 100 x 10 x 2.5 = 250,000 units of volume, and expecting one misread
    # of a gene in it means a density of 1/250,000, or 4e-6.
    #
    # You can give one number for all genes, or a dictionary with a 'default' key
    # plus per gene overrides for genes you know are dirty:
    #   {'default': 1e-6, 'Plp1': 1e-4}
    # A bare number is turned into {'default': <number>} for you.
    #
    # Raising it makes the background a stronger competitor, so more spots are
    # called misreads. Lowering it pushes more spots into cells.
    "MisreadDensity": 0.00001,


    # Each gene gets its own background density, worked out from the reads of that
    # gene that end up in the background. The prior puts that density at the
    # MisreadDensity value, the same for every gene, and rRho decides how firmly
    # it is held there. What matters is the value of rRho compared to the number
    # of background reads the gene has. Set it very large and every gene keeps the
    # MisreadDensity value, so there is a single background level for the whole
    # panel. At 1 rRho has almost no influence, because any gene with real
    # background has far more reads than that, and the data determine the density
    # gene by gene.
    "rRho": 1.0,


    # TBD: cell_centroid_prior has no effect at the moment.
    "cell_centroid_prior": 10,


    # TBD: cell_cov_prior has no effect at the moment.
    "cell_cov_prior": 10,


    # A small number added to the expected counts, so that one stray spot cannot
    # rule out a whole class.
    #
    # Class definitions contain exact zeros: if a class never expresses a gene,
    # its expected count for that gene is 0. A single spot of that gene landing
    # in such a cell would then be impossible under that class, and the class
    # would be eliminated outright, on the strength of one spot that could easily
    # be a misread. This keeps the expectation just above zero so that cannot
    # happen. Leave it alone unless you know why you are changing it.
    "SpotReg": 0.1,


    # How many candidate cells each spot is scored against. The spot is compared
    # with its nearest cells and, on top of those, with the background, so at
    # nNeighbors=6 there are seven options: six cells or the background, ie a
    # misread. Raising it lets a spot reach a cell further away, at the cost of
    # more work per iteration. Lowering it is faster but a spot near a cell
    # boundary may not see the cell it actually came from.
    "nNeighbors": 6,


    # Write the results to disk as tsv files. Without an output_path they land in
    # a 'pciSeq' folder in your system temp directory.
    "save_data": True,


    # Extra logging: how long each step took and a per iteration breakdown of
    # which cell and which spot moved most. None of it changes the result and all
    # of it is cheap, so it is reasonable to leave on when you care about a run.
    "verbose": False,


    # elbo_per_step is a diagnostic for developers. It scores the ELBO either side
    # of the named steps and logs the change, which shows whether a step is
    # improving the fit or working against it. An empty list is off and free.
    #
    # Valid names: geneCount_upd, rho_upd, eta_upd, theta_upd, gamma_upd,
    # cell_to_cellType, dalpha_upd, spots_to_cell. Pass "all" for the lot,
    # but note the ELBO is expensive: scoring every step ran 4.8x slower on
    # silver 180, 1998s against 420s for the same 10 iterations.
    "elbo_per_step": [],


    # Where the results go. 'default' means your system temp folder.
    "output_path": "default",


    # Cell radius, in the same units as the spot coordinates. Leave it at None
    # and the mean radius across all segmented cells is used, which is usually
    # what you want. Set it only if the segmentation gives sizes you do not
    # trust.
    "cell_radius": None,


    # How the class prior is handled. Cell classes are not equally common, and
    # the prior tilts the scores so that, all else being equal, a cell is called
    # a common class rather than a rare one.
    #   'uniform'  the weights you give below are used unchanged all the way
    #              through.
    #   'weighted' the weights of the real classes are re-estimated each
    #              iteration from how many cells currently look like each class,
    #              so the prior adapts to your tissue. The Zero class weight is
    #              held fixed and is not re-estimated.
    "cell_type_prior": "uniform",


    # cell_type_weights sets the starting class weights, as probabilities summing
    # to 1. Name the classes you want to pin and the rest share what is left
    # equally: {"Zero": 0.5} gives Zero half and splits the other half evenly over
    # the real classes. None gives every class the same weight. A name that does
    # not match a class in your reference is ignored with a warning, and there is
    # no special 'default' key here.
    #
    # Zero is the class that expects no expression. It takes the cells that are
    # effectively empty: debris, fragments left by the segmentation, and cells
    # whose markers are not on the panel. Giving it a decent share of the prior
    # keeps those cells off the real classes.
    "cell_type_weights": {"Zero": 0.5},



    # *******************************************************************************
    # 3D settings
    # *******************************************************************************

    # Physical size of a voxel as [x, y, z], in whatever unit you like as long as
    # all three use the same one. It matters when the z step is coarser than the
    # xy pixel, which it usually is: a cell that looks spherical in voxels is
    # really squashed, and without this the distances along z are wrong.
    # For 0.147 um pixels and a 0.9 um z step, use [0.147, 0.147, 0.9].
    # [1, 1, 1] means isotropic and is right for 2D.
    "voxel_size": [1, 1, 1],


    # Drop cells that appear on a single z plane. These are usually segmentation
    # artefacts rather than real cells. Has no effect on 2D data.
    "remove_flat_cells": True,



    # *******************************************************************************
    # Live viewer, optional
    # *******************************************************************************

    # Watch the cell calling as it runs, in a browser.
    "realtime_viewer": False,


    # Port the viewer serves on.
    "realtime_viewer_port": 5001,


    # Cap on how many cells are drawn, for speed on large sections. None draws
    # all of them.
    "realtime_viewer_max_cells": None,


    # Draw every cell at this radius instead of its own. None uses the real
    # sizes.
    "realtime_viewer_fixed_radius": None,
}


# pciSeq works these out from your data and keeps them in the same dictionary as
# the settings above.
#
# - `is3D`: a stack or a single plane, from the masks you passed.
# - `img_dim`: width, height and number of planes of the label image.
# - `label_map`: the renumbering applied to the cell labels, if any.
RUNTIME_KEYS = ('is3D', 'img_dim', 'label_map')

