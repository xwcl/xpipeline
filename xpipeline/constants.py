from enum import Enum

DQ_BAD_PIXEL = 0b00000001
DQ_SATURATED = 0b00000010
DQ_NOT_OVERLAPPING = 0b00000100
DQ_INTERPOLATED_VALUE = 0b00001000
DQ_BG_ESTIMATE_PIX = 0b00010000
HEADER_KW_INTERPOLATED = "INTRPLTD"

class KlipStrategy(Enum):
    SVD = "svd"
    DOWNDATE_SVD = "downdate_svd"
    COVARIANCE = "covariance"

class ValueFilter(Enum):
    DIFFERENCE_FROM_CURRENT = 'difference_from_current'
    ABSOLUTE = 'absolute'

class CombineOperation(Enum):
    MEAN = 'mean'
    SUM = 'sum'
    MEDIAN = 'median'
    STD = 'std'

class NormalizeToUnit(Enum):
    PEAK = 'peak'
    TOTAL = 'total'

class CompareOperation(Enum):
    GT = 'gt'
    GE = 'ge'
    EQ = 'eq'
    LE = 'le'
    LT = 'lt'
    NE = 'ne'

    @property
    def ascii_operator(self):
        if self.value == 'gt':
            return '>'
        elif self.value == 'ge':
            return '>='
        elif self.value == 'eq':
            return '=='
        elif self.value == 'le':
            return '<='
        elif self.value == 'lt':
            return '<'
        elif self.value == 'ne':
            return '!='
        else:
            return RuntimeError(f"Unknown comparator {self.value}")