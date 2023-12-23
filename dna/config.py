from __future__ import annotations

import os
from typing import Optional, Union, Any
from pathlib import Path

from argparse import Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

_NOT_EXISTS:DictConfig = OmegaConf.create(None)


def load(config_path:Union[str,Path]) -> DictConfig:
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    conf = OmegaConf.load(config_path)
    assert isinstance(conf, DictConfig)
    return conf


def to_conf(value:dict[str,Any]|Namespace|DictConfig=dict()) -> DictConfig:
    if OmegaConf.is_config(value):
        assert isinstance(value, DictConfig)
        return value
    elif isinstance(value, Namespace):
        return OmegaConf.create(vars(value))
    elif isinstance(value, dict):
        return OmegaConf.create(value)
    else:
        raise ValueError(f'invalid configuration: {value}')


def exists(conf:DictConfig, key:str) -> bool:
    return OmegaConf.select(conf, key, default=_NOT_EXISTS) != _NOT_EXISTS
    

def get(conf:DictConfig, key:str, *, default:Optional[object]=None) -> Any:
    return OmegaConf.select(conf, key, default=default)
    

def get_or_insert_empty(conf:DictConfig, key:str) -> Any:
    value = OmegaConf.select(conf, key, default=_NOT_EXISTS)
    if value == _NOT_EXISTS:
        value = OmegaConf.create()
        OmegaConf.update(conf, key, value)
    return value


def get_parent(conf:DictConfig, key:str) -> tuple[Optional[DictConfig], Optional[str], str]:
    last_idx = key.rfind('.')
    if last_idx >= 0:
        parent = OmegaConf.select(conf, key[:last_idx])
        return parent, key[:last_idx], key[last_idx+1:]
    else:
        return (None, None, key)


def update(conf:DictConfig, key:str, value:dict[str,object]|Namespace|DictConfig,
           *,
           ignore_if_exists:bool=False) -> None:
    if ignore_if_exists and exists(conf, key):
        return
        
    values_dict = value
    if isinstance(value, Namespace):
        values_dict = vars(value)
    elif isinstance(value, DictConfig):
        values_dict = dict(value)

    OmegaConf.update(conf, key, value, merge=True)
    
def update_values(conf:DictConfig, values:dict[str,Any]|Namespace|DictConfig, *keys) -> None:
    if isinstance(values, dict):
        values_dict:dict[str,Any] = values
    elif isinstance(values, Namespace):
        values_dict:dict[str,Any] = vars(values)
    elif isinstance(values, DictConfig):
        values_dict:dict[str,Any] = dict(values)
        
    for k, v in values_dict.items():
        if not keys or k in keys:
            OmegaConf.update(conf, k, v, merge=True)
    

def filter(conf:DictConfig, *keys:str) -> DictConfig:
    return OmegaConf.masked_copy(conf, list(keys))

def remove(conf:DictConfig, key:str) -> DictConfig:
    parent, s1, leaf = get_parent(conf, key)
    if parent is not None:
        parent.pop(leaf, None)
        return OmegaConf.create(conf)
    else:
        return conf

def exclude(conf:DictConfig, *keys:str) -> DictConfig:
    conf_dict = dict(conf)
    for k, v in dict(conf).items():
        print(k not in keys)
    
    return OmegaConf.create({k:v for k, v in dict(conf).items() if k not in keys})

        
def to_dict(conf:DictConfig) -> dict[str,Any]:
    return dict(conf)