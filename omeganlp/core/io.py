#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import inspect
import json
import os
import shutil
import tarfile
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Union

from omegaconf import DictConfig, OmegaConf


def _read_json_per_line(file_path, filter_, ignore_exception):
    with open(file_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if filter_ is None or filter_(data):
                    yield data
            except Exception as e:
                if not ignore_exception:
                    raise e


def read_json(
    file_path: str,
    per_line_mode: bool = True,
    filter_: Optional[Callable[[Dict], bool]] = None,
    ignore_exception: bool = False,
) -> Union[Iterable[Dict], Dict]:

    if per_line_mode:
        return _read_json_per_line(file_path, filter_, ignore_exception)
    else:
        with open(file_path) as f:
            try:
                return json.load(f)
            except Exception as e:
                if not ignore_exception:
                    raise e


def dump_json(
    objs: Union[Iterable[Any], Any],
    file_path: str,
    per_line_mode: bool = True,
    ensure_ascii: bool = False,
) -> None:
    with open(file_path, "w") as f:
        if per_line_mode:
            for obj in objs:
                json.dump(obj, f, ensure_ascii=ensure_ascii)
                f.write("\n")
        else:
            json.dump(objs, f, ensure_ascii=ensure_ascii)


def read_vocab_txt(file_path: str) -> List[str]:
    results = []
    with open(file_path) as f:
        for line in f:
            results += [line.strip()]
    return results


def dump_vocab_txt(vocab: Iterable[Union[str, Any]], file_path: str) -> None:
    with open(file_path, "w") as f:
        for v in vocab:
            f.write(v)
            f.write("\n")


def tgz(source_dir: str, output_path: Optional[str] = None) -> str:
    basename = os.path.basename(source_dir)
    if output_path is None:
        output_path = basename + ".tgz"
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=basename)
    return os.path.abspath(output_path)


def untgz(source_path: str, target_dir: str = ".") -> None:
    with tarfile.open(source_path, "r") as tar:
        tar.extractall(path=target_dir)


def is_empty(path: str) -> Optional[bool]:
    if os.path.exist(path):
        files = os.listdir(path)
        return len(files) <= 0


def clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


class OmegaDataDir:
    root = os.environ.get("OMEGA_DATA")
    if root is None:
        default_path = os.path.expanduser("~/.omega")
        import warnings

        warnings.warn(
            "No Environment Variable:OMEGA_DATA Detected! Automatically set OMAGA_DATA by default at {}".format(
                default_path
            )
        )
        del warnings
        os.makedirs(default_path, exist_ok=True)
        root = default_path

    pretrained = os.path.join(root, "pretrained")
    datasets = os.path.join(root, "datasets")
    embeddings = os.path.join(root, "pretrained", "embeddings")
    sota = os.path.join(root, "sota")
    extras = os.path.join(root, "extras")

    @staticmethod
    def parse(path: str):
        if isinstance(path, str) and path.startswith("omega://"):
            return os.path.join(OmegaDataDir.root, path[8:])
        else:
            return path

    @staticmethod
    def parse_config(config: MutableMapping):
        for k, v in config.items():
            if isinstance(v, str):
                _v = OmegaDataDir.parse(v)
            elif isinstance(v, dict):
                _v = OmegaDataDir.parse_mapping(v)
            else:
                _v = v
            config.update({k: _v})
        return config


def check_path(data_dir, name, ignore_exception=False):
    names = os.listdir(data_dir)
    names = [n for n in names if n.split(".")[0] == name]
    if len(names) == 0:
        if not ignore_exception:
            raise Exception(f"No file named {name} in {data_dir}")
    else:
        return os.path.join(data_dir, names[0])


def dump_config(config, config_path):
    assert config_path.endswith(".yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(DictConfig(config), f)


def read_config(config_path):
    assert config_path.endswith(".yaml")
    with open(config_path) as f:
        config = OmegaConf.load(f)
    return config


def find_class(module_class_path):
    module_name = ".".join(module_class_path.split(".")[:-1])
    class_name = module_class_path.split(".")[-1]
    module = importlib.import_module(module_name)
    target_cls = getattr(module, class_name)
    return target_cls


def get_class_fullname(o: Any):
    module = o.__class__.__module__
    name = o.__class__.__name__
    return ".".join([module, name])


def incept_function_args(func):
    args = inspect.getfullargspec(func)
    args = args.args
    return [a for a in args if a != "self"]


def get_attr_method_from_module(module_name, attr_method_name):
    module = importlib.import_module(module_name)
    attr_method = getattr(module, attr_method_name)
    return attr_method


def instantiate_interface(target: str):
    module, *attrs = target.split(".")
    module = importlib.import_module(module)

    interface = module
    for attr in attrs:
        interface = getattr(interface, attr)
    return interface
