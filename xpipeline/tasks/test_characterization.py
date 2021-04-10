import numpy as np
from .characterization import simple_aperture_locations

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
