import numpy as np
from .characterization import simple_aperture_locations, inject_signals, CompanionSpec
from . import improc
from .. import core

def test_simple_aperture_locations():
    r_px = 5
    pa_deg = 0
    diam = 7
    assert np.allclose(
        np.asarray(list(simple_aperture_locations(r_px, pa_deg, diam))),
        [[0, 5], [-5, 0], [0, -5], [5, 0]]
    )
    assert np.allclose(
        np.asarray(list(simple_aperture_locations(r_px, pa_deg, diam, exclude_planet=True))),
        [[-5, 0], [0, -5], [5, 0]]
    )
    assert np.allclose(
        np.asarray(list(simple_aperture_locations(r_px, pa_deg, diam, exclude_planet=True, exclude_nearest=1))),
        [[0, -5]]
    )

def test_inject_signals():
    template = np.zeros((4, 4))
    template[1:3,1:3] = 1
    n_frames = 50
    angles = np.linspace(0, 90, num=n_frames)
    f_frame = improc.f_test(128)
    from skimage.transform import rotate
    # n.b. skimage rotate() uses theta in the opposite sense from
    # improc.quick_derotate, but that's actually what we want here
    # since we're faking data that will then be derotated
    cube = np.asarray([rotate(f_frame, theta) for theta in angles])
    r_px = 40
    theta_deg = 90
    specs = [CompanionSpec(scale=1, r_px=r_px, pa_deg=theta_deg)]

    # NumPy only
    outcube = inject_signals(cube, angles, specs, template)
    derot_result = improc.quick_derotate(outcube, angles)
    assert (derot_result[128//2,128-r_px] - n_frames) < 0.1

    # With Dask
    d_cube = core.dask_array.from_array(cube)
    d_angles = core.dask_array.from_array(angles)
    outcube = inject_signals(d_cube, d_angles, specs, template)
    outcube = outcube.compute()
    derot_result = improc.quick_derotate(outcube, angles)
    assert (derot_result[128//2,128-r_px] - n_frames) < 0.1
