import dask
import dask.array as da
from astropy.io import fits
import numpy as np

@dask.delayed
def compute_components(sky_cube, n_components):
    sky_cube[da.isnan(sky_cube)] = 0.0
    mean_sky_image = da.mean(sky_cube, axis=0)
    planes, rows, cols = sky_cube.shape
    all_real_mtx = (sky_cube - mean_sky_image).reshape((planes, rows * cols)).T  # now cube is rows*cols x planes
    mtx_u, _, _ = da.linalg.svd_compressed(all_real_mtx, k=n_components)  # mtx_u is rows*cols x n_components

    if n_components > mtx_u.shape[1]:
        raise ValueError(f"Couldn't compute {n_components} components from {sky_cube.shape} cube")
    components_cube = mtx_u.T.reshape((n_components, rows, cols))

    return components_cube, mean_sky_image

@dask.delayed
def reconstruct_masked(original_image, mean_image, components_cube, mask):
    '''Reconstruct the masked regions of `original_image` with images from
    `components_cube`
    mask pixels with True are excluded (either by replacing
    with `fill` or ignored in least-squares fit). fit uses only "~mask" pixels
    '''
    meansub_image = original_image - mean_image
    mask_1d = mask.flatten()
    image_subset = meansub_image[mask]
    components_subset = components_cube[:,mask]
    
    x, residuals, rank, s = np.linalg.lstsq(
        fitter.components_[:,~mask_1d].T,
        original_image.flatten()[~mask_1d] - fitter.mean_[~mask_1d],
        rcond=None
    )
    reconstruction = (np.dot(fitter.components_.T, x) + fitter.mean_).reshape(original_image.shape)
    return reconstruction
