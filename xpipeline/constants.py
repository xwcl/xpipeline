from enum import Enum
import distributed.protocol

DQ_BAD_PIXEL = 0b00000001
DQ_SATURATED = 0b00000010
DQ_NOT_OVERLAPPING = 0b00000100
DQ_INTERPOLATED_VALUE = 0b00001000
DQ_BG_ESTIMATE_PIX = 0b00010000
HEADER_KW_INTERPOLATED = "INTRPLTD"

class KlipStrategy(Enum):
    SVD = 1
    DOWNDATE_SVD = 2
    COVARIANCE = 3

class ValueFilter(Enum):
    DIFFERENCE_FROM_CURRENT = 'difference_from_current'
    ABSOLUTE = 'absolute'

class CombineOperation(Enum):
    MEAN = 'mean'
    SUM = 'sum'

# distributed.protocol.register_generic(KlipStrategy)
