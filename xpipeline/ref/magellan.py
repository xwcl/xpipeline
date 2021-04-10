from astropy import units as u
import numpy as np

# From LCO telescope information page
PRIMARY_MIRROR_DIAMETER = 6502.4 * u.mm
PRIMARY_STOP_DIAMETER = 6478.4 * u.mm
SECONDARY_AREA_FRACTION = 0.074
# computed from the above
SECONDARY_DIAMETER = 2 * np.sqrt(((PRIMARY_STOP_DIAMETER / 2)**2) * SECONDARY_AREA_FRACTION)
# from MagAO-X Pupil Definition doc
SPIDERS_OFFSET = 0.34 * u.m
