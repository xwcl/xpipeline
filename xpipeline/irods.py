import fnmatch
import os
import ssl
import threading
import logging

import dask
from irods.session import iRODSSession
from irods.meta import iRODSMeta, AVUOperation
import irods_fsspec
irods_fsspec.register()

log = logging.getLogger(__name__)
_irods_global = threading.local()


def get_session():
    if hasattr(_irods_global, 'session'):
        return _irods_global.session
    try:
        env_file = os.environ['IRODS_ENVIRONMENT_FILE']
    except KeyError:
        env_file = os.path.expanduser('~/.irods/irods_environment.json')

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None)
    ssl_settings = {'ssl_context': ssl_context}
    session = iRODSSession(irods_env_file=env_file, **ssl_settings)
    _irods_global.session = session
    return session


def glob(root_collection_path, pattern):
    '''Match data objects (files) with names matching
    `pattern` that are members of the collection identified
    by `root_collection_path`
    '''
    session = get_session()
    coll = session.collections.get(root_collection_path)
    return [x for x in coll.data_objects if fnmatch.fnmatchcase(x.name, pattern)]

def _normalize_metadata_dict(metadata_dict):
    out = {}
    for key, val in metadata_dict.items():
        out[key] = str(val)
    return out

def attach_metadata(data_object, metadata_dict):
    metadata_dict = _normalize_metadata_dict(metadata_dict)  # coerce to strs
    # remove any existing avus whose names would collide
    # (note this also turns multiply-valued keys from our metadata_dict into
    # singly valued, which is what we want and expect elsewhere)
    for avu in data_object.metadata.items():
        if avu.name in metadata_dict and avu.value != metadata_dict[avu.name]:
            log.info(f'Removing metadata key {avu.key} from {data_object}')
            data_object.metadata.remove(avu)
    for key in metadata_dict.keys():
        new_avu = iRODSMeta(key, metadata_dict[key])
        log.info(f'Setting metadata key {key}={metadata_dict[key]} on {data_object}')
        data_object.metadata.add(new_avu)

def ensure_collection(collection_path):
    session = get_session()
    collection_path = os.path.normpath(collection_path)
    if not session.collections.exists(collection_path):
        log.info(f'Creating iRODS collection {collection_path}')
        session.collections.create(collection_path)
    else:
        log.info(f'iRODS collection {collection_path} exists')
    return collection_path
