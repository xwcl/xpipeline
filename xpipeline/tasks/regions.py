import numpy as np
import logging
import re
from typing import Union
from .improc import mask_arc, mask_box
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class Circle:
    center_x : float
    center_y : float
    radius_px : float

    def mask(self, shape):
        return mask_arc((self.center_y, self.center_x), shape, from_radius=0, to_radius=self.radius_px)

@dataclass
class Box:
    center_x : float
    center_y : float
    width : float
    height : float
    rotation_deg : float

    def mask(self, shape):
        return mask_box((self.center_y, self.center_x), shape, (self.height, self.width), rotation_deg=self.rotation_deg)

Region = Union[Circle,Box]

REGION_RE_OPTIONS = (
    re.compile(r'^(box)\(([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)$'),
    re.compile(r'^(circle)\(([\d.]+),([\d.]+),([\d.]+)\)$'),
)

def load_file(fh):
    log.debug(f'Loading region from {fh}')
    regions = []
    for line in fh:
        if isinstance(line, bytes):
            line = line.decode('utf8')
        for re_opt in REGION_RE_OPTIONS:
            res = re_opt.match(line)
            if res is not None:
                break
        if res is None:
            log.debug(f'skipping region file line: {line}')
            continue
        groups = res.groups()
        log.debug(groups)
        name = groups[0]
        parts = [float(x) for x in groups[1:]]
        if name == 'box':
            x, y, width, height, rot = parts
            reg = Box(x, y, width, height, rot)
        elif name == 'circle':
            x, y, radius = parts
            reg = Circle(x, y, radius)
        regions.append(reg)
    return regions
        
def make_mask(regions: list[Region], shape: tuple[int,int], mask_regions_as_true:bool=True):
    '''
    Parameters
    ----------
    regions 
    '''
    mask = np.zeros(shape, dtype=bool)
    for region in regions:
        mask |= region.mask(shape)
    if mask_regions_as_true:
        return mask
    else:
        return ~mask
