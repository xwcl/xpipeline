import numpy as np
import logging
import re
from typing import Optional, Union
from .improc import mask_arc, mask_box
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class Circle:
    radius_px : float
    center_x : float
    center_y : float
    text : Optional[str] = None

    def mask(self, shape):
        return mask_arc((self.center_y, self.center_x), shape, from_radius=0, to_radius=self.radius_px)

@dataclass
class Box:
    center_x : float
    center_y : float
    width : float
    height : float
    rotation_deg : float
    text : Optional[str] = None

    def mask(self, shape):
        return mask_box((self.center_y, self.center_x), shape, (self.height, self.width), rotation_deg=self.rotation_deg)

Region = Union[Circle,Box]

REGION_RE_OPTIONS = (
    re.compile(r'^(box)\(([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)(?:\s+#\s+text=\{(.+)\})?$'),
    # "circle(372,553,86.188632) # text={a}"
    re.compile(r'^(circle)\(([\d.]+),([\d.]+),([\d.]+)\)(?:\s+#\s+text=\{(.+)\})?$'),
)

def load_file(fh) -> list[Region]:
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
        parts = groups[1:]
        if name == 'box':
            x, y, width, height, rot, maybe_text = parts
            reg = Box(center_x=float(x), center_y=float(y), width=float(width), height=float(height), rotation_deg=float(rot), text=maybe_text)
        elif name == 'circle':
            x, y, radius, maybe_text = parts
            reg = Circle(center_x=float(x), center_y=float(y), radius_px=float(radius), text=maybe_text)
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
