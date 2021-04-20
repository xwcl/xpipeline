``xp local_klip``
=================

Apply Karhunen-Loève Image Projection starlight subtraction and angular differential imaging to either a collection of FITS files representing a sequence of ADI observations, or to a cube of observations with corresponding angle information in a multi-extension FITS file.

Differs from :doc:`klip` in that:

* there is no attempt to distribute computations over cores/nodes (except as provided automatically by NumPy's chosen linear algebra library)
* initial decomposition takes place with the default :py:func:`numpy.linalg.svd` rather than Halko approximate SVD (from Dask). (This may change in the future.)

.. argparse::
    :ref: xpipeline.commands.local_klip._docs_args
    :prog: xp local_klip
    :passparser:
