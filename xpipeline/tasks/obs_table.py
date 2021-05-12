import numpy as np
import warnings
import dask
import pandas as pd
from astropy.io import fits

IGNORED_HEADER_KEYWORDS = ("EXTEND", "COMMENT", "EXTNAME", "SIMPLE", "XTENSION")


@dask.delayed
def headers_to_dict(hdulist, original_name, observation_date_key):
    result = {"original_name": original_name}
    for hdu in hdulist:
        for key in hdu.header:
            if key in IGNORED_HEADER_KEYWORDS:
                continue
            result[key] = hdu.header.get(key)

    if observation_date_key in result:
        result[observation_date_key] = np.datetime64(result[observation_date_key])
    else:
        result[observation_date_key] = None
    return result


@dask.delayed
def construct_observations_table(names_to_hdulists, observation_date_key="DATE-OBS"):
    results = dask.compute(
        *[
            headers_to_dict(hdul, name, observation_date_key)
            for name, hdul in names_to_hdulists.items()
        ]
    )
    all_columns = [observation_date_key]
    for r in results:
        for key in r:
            if key not in all_columns:
                all_columns.append(key)

    # construct table
    df = pd.DataFrame(index=np.arange(0, len(results)), columns=all_columns)
    for idx, frame_data in enumerate(results):
        df.loc[idx] = [frame_data.get(col) for col in all_columns]
    df.sort_values(observation_date_key, inplace=True)
    return df.reset_index(drop=True)  # idx <=> date
