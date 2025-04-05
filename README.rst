magtomo
=======
magtomo is a python package for performing scalar, vector and orientation tomography.

Installation
------------
magtomo can be installed through pip:

.. code-block:: console

   (.venv)$ pip install magtomo

Examples
--------
Examples of scalar, vector and orientation reconstructions can be found in the magtomo.examples() module. To simulate
the process, projections are first calculated

.. code-block:: python

    # scalar structure projections
    projections = radon(struct, angles)
    # vector or orientation field projections
    exp = Experiment(magnetization=struct, rotations=rot, pol=pol)
    exp.calculate_sinogram()
    projections = exp.sinogram

the projections are then used as input for the reconstruction process

.. code-block:: python

    # inverse of scalar projections
    recons = inv_radon(projections, angles)
    # reconstruction from vector or orientation projections
    rec = Reconstruction(initial_guess, rotations=rot, projections=projections,
                         pol=pol, mask=mask)
    rec.reconstruct()
    recons = rec.magnetization


Documentation
-------------
Comprehensive documentation is available online at `readthedocs <https://magtomo.readthedocs.io/en/latest/index.html>`_.