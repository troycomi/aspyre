Getting Started
===============

The following top-level scripts are currently available in the ``scripts`` folder to run different stages of the Cryo-EM data pipeline.
To run these, make sure you have activated the proper conda environment for ASPyRE (see :doc:`installation` for installation instructions).

1. Particle-Picking
*******************

The ``apple.py`` script takes in a folder of one or more ``*.mrc`` files, picks particles using the Apple-Picker algorithm described at
:cite:`DBLP:journals/corr/abs-1802-00469`, and generates ``*.star`` files, one for each ``*.mrc`` file processed, at a folder location
specified by the ``-o`` flag.

For example, to run the script on sample data included in ASPyRE (a single ``falcon_2012_06_12-14_33_35_0.mrc`` file provided from the 5.3 GB
`Beta-galactosidase Falcon-II micrographs EMPIAR dataset <https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/>`_):

::

    cd scripts
    mkdir apple_output
    python apple.py ../tests/saved_test_data/mrc_files -o apple_output

2. Simulation
*************

The ``simulation.py`` script generates simulated Gaussian volumes as source of (noisy) images, runs the ASPyRE pipeline to determine the estimated
mean volume and estimated covariance on the mean volume, and runs evaluations on these estimated quantities (against the `true` values which
we know from the simulation).

::

    cd scripts
    python simulation.py

3. Processing Starfiles
***********************

The ``starfile.py`` script takes in a ``*.star`` file, processes the images (*.mrcs files) found in the starfile, and runs the ASPyRE pipeline
to determine the estimated mean volume and estimated covariance on the mean volume. No results are saved by the script, but this script is
a good place to look for a `real-life` example of running the ASPyRE pipeline.

For example, to run the script on sample data included in ASPyRE:

::

    cd scripts
    python starfile.py --starfile ../tests/saved_test_data/starfile.star --ignore_missing_files --pixel_size 1.338 -L 8 --cg_tol 0.2

.. note::

    Pay special attention to the flags specified in the example above. The ``--ignore_missing_files`` flag ignores any *.mrcs files
    referenced by the starfile but not found in the filesystem (as is the case with sample data provided in ASPyRE). The ``-L 8``
    flag down-samples images to 8x8 pixels (needed otherwise you may run out of memory, and/or the script may take way too long to execute).
    The ``cg_tol`` flag is explained below.

The ``starfile.py`` script also provides an example of how to override configuration values (that are read from ``config.json`` file
provided in the package. In the above example, ``cg_tol=0.2`` sets very liberal (and unrealistic) limits on optimization convergence
tolerance, which is needed for such a small dataset. For real datasets, you would typically not want to override `cg_tol`.