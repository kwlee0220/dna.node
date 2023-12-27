from __future__ import annotations


from typing import Optional, TypeVar
from collections.abc import Callable
import logging

from datetime import datetime, timezone
from pathlib import Path


def datetime2utc(dt: datetime) -> int:
    # 'timestamp()'를 호출하면 자동적으로 UTC timezone의 값을 반환하기 때문에
    # 별도의 timezone을 고려할 필요가 없다.
    return round(dt.timestamp() * 1000)

def utc2datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)

def datetime2str(dt: datetime, *, include_millis=False) -> str:
    if include_millis:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_seconds() -> float:
    return utc_now_datetime().timestamp()

def utc_now_millis() -> int:
    return round(utc_now_seconds() * 1000)


def _parse_keyvalue(kv:str) -> tuple[str,str]:
    pair:list[str] = kv.split('=')
    if len(pair) == 2:
        return (pair[0], pair[1])
    else:
        return pair, None

def parse_query(query: str) -> dict[str,str]:
    if not query or len(query) == 0:
        return dict()
    return dict([_parse_keyvalue(kv) for kv in query.split('&')])

def get_first_param(args: dict[str,object], key: str, def_value=None) -> object:
    value = args.get(key)
    return value[0] if value else def_value

def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1

def find_track_index(track_id, tracks):
    return next((idx for idx, track in enumerate(tracks) if track[idx].id == track_id), None)


def gdown_file(url:str, file: Path, force: bool=False):
    if isinstance(file, str):
        file = Path(file)
        
    if force:
        file.unlink()

    if not file.exists():
        # create an empty 'weights' folder if not exists
        file.parent.mkdir(parents=True, exist_ok=True)

        import gdown
        gdown.download(url, str(file.resolve().absolute()), quiet=False)

def initialize_logger(logger_conf_file: Optional[str]=None):
    import yaml
    import logging.config
    
    if logger_conf_file is None:
        import pkgutil
        data = pkgutil.get_data('conf', 'logger.yaml')
        if data is None:
            raise ValueError(f"Cannot read the default logger configuration: logger.yaml")
        logger_conf_text = data.decode('utf-8')
    else:
        with open(logger_conf_file, 'rt') as f:
            logger_conf_text = f.read()
    logger_conf = yaml.safe_load(logger_conf_text)
    logging.config.dictConfig(logger_conf)
    
def sub_logger(logger:Optional[logging.Logger], suffix:str) -> Optional[logging.Logger]:
    return logger.getChild(suffix) if logger else None
        
        
def has_method(obj, name:str) -> bool:
    method = getattr(obj, name, None)
    return callable(method) if method else False


T = TypeVar("T")
def get_or_else(value:T, else_value:T|Callable[[],T]) -> T:
    if value:
        return value
    else:
        return else_value() if callable(else_value) else else_value