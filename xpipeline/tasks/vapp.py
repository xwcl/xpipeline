import numpy as np
from . import improc


def make_dark_hole_masks(shape, owa_px, offset_px, psf_rotation_deg):
    '''Generate dark hole masks for a gvAPP-180 using OWA, center
    offset, and rotation

    Parameters
    ----------
    shape : tuple
    owa_px : float
        Outer working angle measured in pixels from the *shifted* center
    offset_px : float
        Center location shift in -X for left and +X for right (adjusted
        by `psf_rotation_deg`)
    psf_rotation_deg : float
        Amount by which the flat edges of the masks (and the offset
        from center) are rotated E of N (CCW when 0,0 at lower left)
    
    Returns
    -------
    left_mask : np.ndarray
        Mask that is True within the dark hole that lies on the left
        side when `psf_rotation_deg` == 0
    right_mask : np.ndarray
        Mask that is True within the dark hole that lies on the right
        side when `psf_rotation_deg` == 0
    '''
    radius_px = owa_px + offset_px
    right_overall_rotation_radians = np.deg2rad(-psf_rotation_deg + 90)
    right_offset_theta = np.deg2rad(psf_rotation_deg)

    left_overall_rotation_radians = np.deg2rad(-90 + -psf_rotation_deg)
    left_offset_theta = np.deg2rad(-90 - psf_rotation_deg)
    ctr_x, ctr_y = (shape[1] - 1) / 2, (shape[0] - 1) / 2
    right_mask = improc.mask_arc(
        (ctr_x + offset_px * np.cos(right_offset_theta), ctr_y + offset_px * np.sin(right_offset_theta)),
        shape,
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=right_overall_rotation_radians,
    )
    left_mask = improc.mask_arc(
        (ctr_x + offset_px * np.cos(left_offset_theta), ctr_y + offset_px * np.sin(left_offset_theta)),
        shape,
        from_radius=0,
        to_radius=radius_px,
        from_radians=0,
        to_radians=np.pi,
        overall_rotation_radians=left_overall_rotation_radians,
    )
    return left_mask, right_mask


def mask_along_angle(shape, deg_e_of_n):
    '''Generate complementary masks for a gvAPP-180 by dividing the
    image along a line through the center at an angle `deg_e_of_n`

    Parameters
    ----------
    shape : tuple
    deg_e_of_n : float
        Amount by which the dividing line is rotated E of N
        (CCW when 0,0 at lower left)
    
    Returns
    -------
    left_half : np.ndarray
        Mask that is True for pixels left of the dividing line
        when `deg_e_of_n` == 0
    right_half : np.ndarray
        Mask that is True for pixels right of the dividing line
        when `deg_e_of_n` == 0
    '''
    _, theta = improc.polar_coords(improc.center(shape), shape)
    left_half = (
        (theta < np.deg2rad(-90 + deg_e_of_n)) |
        (
            (theta > np.deg2rad(90 + deg_e_of_n)) & (theta > 0)
        )
    )
    right_half = ~left_half
    return left_half, right_half
