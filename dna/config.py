from __future__ import annotations

import os
from typing import Optional, Union
from pathlib import Path

from argparse import Namespace
from omegaconf import OmegaConf

_NOT_EXISTS = OmegaConf.create(None)


def load(config_path:Union[str,Path]) -> OmegaConf:
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    return OmegaConf.load(config_path)


def to_conf(value:dict[str,object]|Namespace|OmegaConf=dict()) -> OmegaConf:
    if OmegaConf.is_config(value):
        return value
    elif isinstance(value, Namespace):
        return OmegaConf.create(vars(value))
    elif isinstance(value, dict):
        return OmegaConf.create(value)
    else:
        raise ValueError(f'invalid configuration: {value}')


def exists(conf:OmegaConf, key:str) -> bool:
    return OmegaConf.select(conf, key, default=_NOT_EXISTS) != _NOT_EXISTS
    

def get(conf:OmegaConf, key:str, *, default:Optional[object]=None) -> object:
    return OmegaConf.select(conf, key, default=default)
    

def get_or_insert_empty(conf:OmegaConf, key:str) -> object:
    value = OmegaConf.select(conf, key, default=_NOT_EXISTS)
    if value == _NOT_EXISTS:
        value = OmegaConf.create()
        OmegaConf.update(conf, key, value)
    return value


def get_parent(conf:OmegaConf, key:str) -> tuple[OmegaConf, str, str]:
    last_idx = key.rfind('.')
    if last_idx >= 0:
        parent = OmegaConf.select(conf, key[:last_idx])
        return parent, key[:last_idx], key[last_idx+1:]
    else:
        return (None, None, key)


def update(conf:OmegaConf, key:str, value:dict[str,object]|Namespace|OmegaConf,
           *,
           ignore_if_exists:bool=False) -> None:
    if ignore_if_exists and exists(conf, key):
        return
        
    values_dict = value
    if isinstance(value, Namespace):
        values_dict = vars(value)
    elif isinstance(value, OmegaConf):
        values_dict = dict(value)

    OmegaConf.update(conf, key, value, merge=True)
    
def update_values(conf:OmegaConf, values:dict[str,object]|Namespace|OmegaConf, *keys) -> None:
    values_dict = values
    if isinstance(values, Namespace):
        values_dict = vars(values)
    elif isinstance(values, OmegaConf):
        values_dict = dict(values)
        
    for k, v in values_dict.items():
        if not keys or k in keys:
            OmegaConf.update(conf, k, v, merge=True)
    

def filter(conf:OmegaConf, *keys:list[str]) -> OmegaConf:
    return OmegaConf.masked_copy(conf, keys)

def remove(conf:OmegaConf, key:str) -> OmegaConf:
    parent, s1, leaf = get_parent(conf, key)
    if parent is not None:
        parent.pop(leaf, None)
        return OmegaConf.create(conf)
    else:
        return conf

def exclude(conf:OmegaConf, *keys:list[str]) -> OmegaConf:
    conf_dict = dict(conf)
    for k, v in dict(conf).items():
        print(k not in keys)
    
    return OmegaConf.create({k:v for k, v in dict(conf).items() if k not in keys})

        
def to_dict(conf:OmegaConf) -> dict[str,object]:
    return OmegaConf.to_container(conf)