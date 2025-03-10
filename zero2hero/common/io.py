import json
import logging
import os
import pickle
import shutil
from io import BytesIO, StringIO
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
import yaml


def dump(obj,
         file = None,
         file_format = None,
         file_client_args = None,
         backend_args = None,
         **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None'
            )
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.'
            )

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args = backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def makedirs(path, verbose = False):
    os.makedirs(path, exist_ok = True)
    if verbose:
        print(f"创建文件夹： {path}")


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024 ** 2)
    return size_in_mb


def save_file(data, filename, append_to_json = True, verbose = True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with open(filename, "wb") as fopen:
            pickle.dump(data, fopen)
    elif file_ext == ".npy":
        with open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys = True) + "\n")
                fopen.flush()
        else:
            with open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys = True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    elif file_ext == ".pt":
        torch.save(data, filename)
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load_file(filename, mmap_mode = None, verbose = True, allow_pickle = False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with open(filename, "r") as fopen:
            data = fopen.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding = "latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with open(filename, "rb") as fopen:
                    data = np.load(
                        fopen,
                        allow_pickle = allow_pickle,
                        encoding = "latin1",
                        mmap_mode = mmap_mode,
                    )
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without g_pathmgr"
                )
                data = np.load(
                    filename,
                    allow_pickle = allow_pickle,
                    encoding = "latin1",
                    mmap_mode = mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without  Trying without mmap")
                with open(filename, "rb") as fopen:
                    data = np.load(fopen, allow_pickle = allow_pickle, encoding = "latin1")
        else:
            with open(filename, "rb") as fopen:
                data = np.load(fopen, allow_pickle = allow_pickle, encoding = "latin1")
    elif file_ext == ".json":
        with open(filename, "r") as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with open(filename, "r") as fopen:
            data = yaml.load(fopen, Loader = yaml.FullLoader)
    elif file_ext == ".csv":
        with open(filename, "r") as fopen:
            data = pd.read_csv(fopen)
    elif file_ext == '.pt':
        data = torch.load(filename)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data
