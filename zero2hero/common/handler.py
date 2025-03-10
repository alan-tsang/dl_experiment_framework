import json
import pickle
import yaml
import io
from pathlib import Path

class BaseHandler:
    """Base handler with text/binary mode flag"""
    str_like = True  # True for text-based formats (json/yaml), False for binary

    @classmethod
    def dump_to_str(cls, obj, **kwargs):
        raise NotImplementedError

    @classmethod
    def dump_to_fileobj(cls, obj, file, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_str(cls, string, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_fileobj(cls, file, **kwargs):
        raise NotImplementedError

class JsonHandler(BaseHandler):
    str_like = True

    @classmethod
    def dump_to_str(cls, obj, **kwargs):
        return json.dumps(obj, **kwargs)

    @classmethod
    def dump_to_fileobj(cls, obj, file, **kwargs):
        if isinstance(file, io.TextIOBase):
            json.dump(obj, file, **kwargs)
        else:
            # Handle binary file objects
            json.dump(obj, io.TextIOWrapper(file, encoding='utf-8'), **kwargs)

    @classmethod
    def load_from_str(cls, string, **kwargs):
        return json.loads(string, **kwargs)

    @classmethod
    def load_from_fileobj(cls, file, **kwargs):
        if isinstance(file, io.TextIOBase):
            return json.load(file, **kwargs)
        else:
            return json.load(io.TextIOWrapper(file, encoding='utf-8'), **kwargs)

class YamlHandler(BaseHandler):
    str_like = True

    @classmethod
    def dump_to_str(cls, obj, **kwargs):
        return yaml.safe_dump(obj, **kwargs)

    @classmethod
    def dump_to_fileobj(cls, obj, file, **kwargs):
        if isinstance(file, io.TextIOBase):
            yaml.safe_dump(obj, stream=file, **kwargs)
        else:
            yaml.safe_dump(obj, stream=io.TextIOWrapper(file, encoding='utf-8'), **kwargs)

    @classmethod
    def load_from_str(cls, string, **kwargs):
        return yaml.safe_load(string, **kwargs)

    @classmethod
    def load_from_fileobj(cls, file, **kwargs):
        if isinstance(file, io.TextIOBase):
            return yaml.safe_load(file, **kwargs)
        else:
            return yaml.safe_load(io.TextIOWrapper(file, encoding='utf-8'), **kwargs)

class PickleHandler(BaseHandler):
    str_like = False  # Binary format

    @classmethod
    def dump_to_str(cls, obj, **kwargs):
        raise NotImplementedError("Pickle cannot be safely dumped to string")

    @classmethod
    def dump_to_fileobj(cls, obj, file, **kwargs):
        pickle.dump(obj, file, **kwargs)

    @classmethod
    def load_from_str(cls, string, **kwargs):
        raise NotImplementedError("Pickle cannot be safely loaded from string")

    @classmethod
    def load_from_fileobj(cls, file, **kwargs):
        return pickle.load(file, **kwargs)

# File handlers registry
file_handlers = {
    'json': JsonHandler,
    'yaml': YamlHandler,
    'yml': YamlHandler,
    'pkl': PickleHandler,
    'pickle': PickleHandler
}

def get_file_backend(uri, backend_args=None):
    """Get appropriate storage backend based on URI scheme"""
    # 实际实现应使用 fsspec 等库处理多后端存储
    # 这里演示本地文件系统实现
    class LocalBackend:
        @staticmethod
        def put(data, dst):
            if isinstance(data, str):
                mode = 'w'
                encoding = 'utf-8'
            else:
                mode = 'wb'
                encoding = None
            with open(dst, mode, encoding=encoding) as f:
                f.write(data)

        @staticmethod
        def get(src):
            with open(src, 'rb') as f:
                return f.read()

    return LocalBackend()