from enum import Enum
import distributed.protocol

DQ_BAD_PIXEL = 0b00000001
DQ_SATURATED = 0b00000010
DQ_NOT_OVERLAPPING = 0b00000100
DQ_INTERPOLATED_VALUE = 0b00001000
DQ_BG_ESTIMATE_PIX = 0b00010000
HEADER_KW_INTERPOLATED = "INTRPLTD"

class KlipStrategy(Enum):
    SVD = 'svd'  # TODO
    COMPRESSED_SVD = 'compressed_svd'  # TODO
    DOWNDATE_SVD = 'downdate_svd'
    DOWNDATE_COMPRESSED_SVD = 'downdate_compressed_svd'  # TODO
    COVARIANCE = 'covariance'  # TODO
    COVARIANCE_TOP_K = 'covariance_top_k'  # TODO

# distributed.protocol.register_generic(KlipStrategy)
