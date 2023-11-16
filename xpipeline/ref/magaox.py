from functools import partial
from ..tasks import detector

def flag_saturation(hdul, **kwargs):
    print(hdul)
    return detector.flag_saturation(hdul, saturation_level=60_000, **kwargs)