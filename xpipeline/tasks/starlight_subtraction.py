from dataclasses import dataclass
import numpy as np
import dask.array as da
from .. import core, utils
from . import learning, improc


import logging
log = logging.getLogger(__name__)


def drop_idx_range_cols(arr, min_excluded_idx, max_excluded_idx):
    '''Note exclusive upper bound: [min_excluded_idx, max_excluded_idx)'''
    xp = core.get_array_module(arr)
    rows, cols = arr.shape
    n_drop = max_excluded_idx - min_excluded_idx
    out_shape = (rows, cols - n_drop)
    out = xp.empty(out_shape, dtype=arr.dtype)
    # L | |  R

    # L
    out[:,:min_excluded_idx] = arr[:,:min_excluded_idx]
    # R
    out[:,min_excluded_idx:] = arr[:,max_excluded_idx:]
    return out


def mean_subtract_vecs(image_vecs):
    xp = core.get_array_module(image_vecs)
    mean_vec = xp.average(image_vecs, axis=1)
    image_vecs_meansub = image_vecs - mean_vec[:,core.newaxis]
    return image_vecs_meansub, mean_vec


class Decomposer:
    def __init__(self, image_vecs, n_modes):
        self.image_vecs = image_vecs
        self.meansub_image_vecs, self.mean_vec = mean_subtract_vecs(image_vecs)
        self.n_modes = n_modes
        self.xp = core.get_array_module(image_vecs)
        self.idxs = self.xp.arange(self.meansub_image_vecs.shape[1])
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        raise NotImplementedError()

class SVDDecomposer(Decomposer):
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        ref = drop_idx_range_cols(self.meansub_image_vecs, min_excluded_idx, max_excluded_idx)
        u, s, v = learning.generic_svd(ref, n_modes=self.n_modes)
        return u[:,:self.n_modes]


class MinimalDowndateSVDDecomposer(Decomposer):
    def __init__(self, image_vecs, n_modes, extra_modes=1):
        super().__init__(image_vecs, n_modes)
        self.n_modes = n_modes
        self.mtx_u, self.diag_s, self.mtx_v = learning.generic_svd(self.meansub_image_vecs, n_modes=n_modes+extra_modes)
        self.idxs = self.xp.arange(image_vecs.shape[1])
    def eigenimages(self, min_excluded_idx, max_excluded_idx):
        new_u, new_s, new_v = learning.minimal_downdate(
            self.mtx_u,
            self.diag_s,
            self.mtx_v,
            min_col_to_remove=min_excluded_idx,
            max_col_to_remove=max_excluded_idx
        )
        return new_u[:,:self.n_modes]

def klip_frame(target, decomposer, exclude_idx_min, exclude_idx_max):
    eigenimages = decomposer.eigenimages(exclude_idx_min, exclude_idx_max)
    meansub_target = target - decomposer.mean_vec
    return meansub_target - eigenimages @ (eigenimages.T @ meansub_target)


def klip_to_modes(image_vecs, decomp_class, n_modes, exclude_nearest=0):
    '''
    Parameters
    ----------
    image_vecs : array (m, n)
        a series of n images arranged into columns of m pixels each
        and mean subtracted such that each pixel-series mean is 0
    decomp_class : Decomposer subclass
        Must accept image_vecs, n_modes, solver in __init__ and implement
        eigenimages() method
    n_modes : int
        Rank of low-rank decomposition
    exclude_nearest : int
        In addition to excluding the current frame, exclude
        this many adjacent frames. (Note the first and last
        few frames won't exclude the same number of frames;
        they will just go to the ends of the dataset.)
    '''
    xp = core.get_array_module(image_vecs)
    _, n_frames = image_vecs.shape

    output = xp.zeros_like(image_vecs)
    idxs = xp.arange(image_vecs.shape[1])
    decomposer = decomp_class(image_vecs, n_modes)
    for i in range(image_vecs.shape[1]):
        output[:,i] = klip_frame(
            image_vecs[:,i],
            decomposer,
            exclude_idx_min=max(i - exclude_nearest, 0),
            exclude_idx_max=min(i+1+exclude_nearest, n_frames)
        )
    return output

def klip_chunk(image_vecs_meansub, mtx_u0, diag_s0, mtx_v0, k_klip, exclude_nearest_n_frames):
    if image_vecs_meansub.shape == (0, 0):
        # called by dask to infer dtype
        return np.zeros_like(image_vecs_meansub)
    log.debug(f"Klipping a chunk {image_vecs_meansub.shape=}")
    n_frames = image_vecs_meansub.shape[1]
    total_n_frames = mtx_v0.shape[0]
    output = np.zeros_like(image_vecs_meansub)
    for i in range(n_frames):
        new_u, new_s, new_v = learning.minimal_downdate(
            mtx_u0,
            diag_s0,
            mtx_v0,
            min_col_to_remove=max(i - exclude_nearest_n_frames, 0),
            max_col_to_remove=min(i+1+exclude_nearest_n_frames, total_n_frames),
        )
        eigenimages = new_u[:,:k_klip]
        meansub_target = image_vecs_meansub[:,i]
        output[:,i] = meansub_target - eigenimages @ (eigenimages.T @ meansub_target)
    mem_mb = utils.get_memory_use_mb()
    log.debug(f'klipped! {mem_mb} MB RAM in use')
    return output

def klip_mtx(image_vecs, k_klip: int, exclude_nearest_n_frames: int):
    xp = core.get_array_module(image_vecs)

    output = xp.zeros_like(image_vecs)
    total_n_frames = image_vecs.shape[1]
    idxs = xp.arange(total_n_frames)
    # extra modes:
    # to remove 1 image vector by downdating, the initial decomposition
    # should retain k_klip + 1 singular value triplets.
    initial_k = k_klip + exclude_nearest_n_frames + 1
    log.debug(f'{image_vecs.shape=}, {k_klip=}, {exclude_nearest_n_frames=}, {initial_k=}')
    if image_vecs.shape[0] < initial_k or image_vecs.shape[1] < initial_k:
        raise ValueError(f"Number of modes requested exceeds dimensions of input")
    image_vecs_meansub, mean_vec = mean_subtract_vecs(image_vecs)
    mtx_u0, diag_s0, mtx_v0 = learning.generic_svd(image_vecs, initial_k)
    log.debug(f'{image_vecs_meansub.shape=}, {mtx_u0.shape=}, {diag_s0.shape=}, {mtx_v0.shape=}')
    if xp is da:
        output = image_vecs_meansub.map_blocks(
            klip_chunk,
            mtx_u0,
            diag_s0,
            mtx_v0,
            k_klip,
            exclude_nearest_n_frames,
            dtype=image_vecs_meansub.dtype
        )
    else:
        output = klip_chunk(image_vecs_meansub, mtx_u0, diag_s0, mtx_v0, k_klip, exclude_nearest_n_frames)
    return output

def make_good_pix_mask(shape, inner_radius=None, outer_radius=None, center=None, existing_mask=None):
    if existing_mask is not None:
        if existing_mask.shape != shape:
            raise ValueError(f"Cannot mix {shape=} and {existing_mask.shape=}")
        mask = existing_mask.copy()
    else:
        mask = np.ones(shape, dtype=bool)

    if center is None:
        center = (shape[1] - 1) / 2, (shape[0] - 1) / 2

    if any(map(lambda x: x is None, (outer_radius, inner_radius))):
        rho, phi = improc.polar_coords(center, shape)
        if inner_radius is not None:
            mask &= rho >= inner_radius
        if outer_radius is not None:
            mask &= rho <= outer_radius

    return mask
