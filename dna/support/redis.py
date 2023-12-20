from typing import Any, Iterable, Iterator, KeysView, Optional, TypeVar
from collections import abc
from urllib.parse import urlparse, parse_qs

import redis

_REMOVED_MARK = '_$REMOVED$_'

RedisType:TypeVar = str|bytes|int|float


def _to_str(value:bytes|Iterable[bytes]) -> str:
    if value is None:
        return None
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    else:
        return [_to_str(v) for v in value]
    
    
def connect(redis_url:str) -> redis.Redis:
    result = urlparse(redis_url)
    
    kwargs = dict()
    if result.query:
        def unwrap(v:list[str]) -> str|list[str]:
            return v if len(v) > 1 else v[0]
        kwargs = {k:unwrap(vl) for k, vl in parse_qs(result.query).items()}
    if result.path and len(result.path) > 0:
        kwargs['db'] = int(result.path[1:])
            
    return redis.Redis(host=result.hostname, port=result.port, **kwargs)
    
class RedisListIterator(abc.Iterator):
    def __init__(self, conn:redis.Redis, key:str, init_pos:int=0) -> None:
        self.conn = conn
        self.key = key
        self.next_pos = init_pos
        
    def __next__(self) -> str:
        ret = self.conn.lindex(self.key, self.next_pos)
        if ret:
            self.next_pos += 1
            return _to_str(ret)
        else:
            raise StopIteration
        

class RedisList(abc.MutableSequence):
    def __init__(self, conn:redis.Redis, key:str, init_list:Optional[Iterable[str]]=None) -> None:
        self.conn = conn
        self.key = key
        if init_list:
            pipe = self.conn.pipeline()
            pipe.delete(self.key)
            pipe.rpush(self.key, *init_list)
            pipe.execute()
            
    def _slice_indexes(self, idx:slice) -> list[Any]:
        length = len(self)
        return range(0, length)[idx]
        
    def __len__(self) -> int:
        return self.conn.llen(self.key)
    
    def __iter__(self) -> Iterator:
        return RedisListIterator(self.conn, self.key, 0)
    
    def __contains__(self, value: Any) -> bool:
        return self.conn.lpos(self.key) is not None
    
    def __getitem__(self, idx:int|slice) -> str|list[str]:
        if isinstance(idx, int):
            return _to_str(self.conn.lindex(self.key, idx))
        else:
            index_list = self._slice_indexes(idx)
            pipe = self.conn.pipeline()
            items = [_to_str(self.conn.lindex(self.key, i)) for i in index_list]
            pipe.execute()
            
            return items
    
    def __setitem__(self, idx:int|slice, value:Any|Iterable) -> None:
        if isinstance(idx, int) and not isinstance(value, str|bytes|float|int):
            self.conn.lset(self.key, idx, value)
        elif isinstance(idx, slice) and isinstance(value, str|bytes|float|int):
            pipeline = self.conn.pipeline()
            length = pipeline.llen(self.key)
            for i in list(range(0, length))[idx]:
                pipeline.lset(self.key, i, value)
            pipeline.execute()
        
    def __delitem__(self, idx:int|slice) -> None:
        if isinstance(idx, int):
            if idx == 0:
                self.conn.lpop(self.key)
            elif idx == -1 or idx == self.conn.llen(self.key) -1:
                self.conn.rpop(self.key)
            else:
                pipeline = self.conn.pipeline()
                pipeline.lset(self.key, idx, _REMOVED_MARK)
                pipeline.lrem(self.key, 1, _REMOVED_MARK)
                pipeline.execute()
        elif isinstance(idx, slice):
            self[idx] = _REMOVED_MARK
            self.conn.lrem(self.key, 0, _REMOVED_MARK)
        else:
            raise ValueError(f'invalid index: {idx}')
    
    def append(self, value: Any) -> None:
        self.conn.rpush(self.key, value)
        
    def insert(self, index: int, value: Any) -> None:
        raise NotImplemented(f'insert with slice')
            
    def __repr__(self) -> str:
        prefix = _to_str(self.conn.lrange(self.key, 0, 8))
        if len(self) > 9:
            return f'[{",".join(prefix)}, ...]'
        else:
            return repr(prefix)


class RedisSet(abc.MutableSet):
    def __init__(self, conn:redis.Redis, key:str, init_set:Optional[Iterable[RedisType]]=None) -> None:
        self.redis = conn
        self.key = key
        if init_set:
            pipe = self.redis.pipeline()
            pipe.delete(self.key)
            pipe.sadd(self.key, *init_set)
            pipe.execute()
        
    def __len__(self) -> int:
        return self.redis.scard(self.key)
    
    def __iter__(self) -> Iterator[RedisType]:
        return self.redis.sscan_iter(self.key)
    
    def __contains__(self, value: RedisType) -> bool:
        return self.redis.sismember(self.key, value)
    
    def add(self, value: RedisType) -> None:
        self.redis.sadd(self.key, value)
    
    def discard(self, value: RedisType) -> None:
        self.redis.srem(self.key, value)
        
    def update(self, values: Iterable[RedisType]) -> None:
        self.redis.sadd(self.key, list(values))


class RedisDict(abc.MutableMapping):
    def __init__(self, conn:redis.Redis, key:str, init_values:Optional[dict[str,Any]]=None) -> None:
        self.redis = conn
        self.key = key
        if init_values:
            pipe = self.redis.pipeline()
            pipe.delete(self.key)
            pipe.hmset(self.key, init_values)
            pipe.execute()
    
    def __getitem__(self, item_key:str) -> str:
        return self.redis.hget(self.key, item_key)
    
    def __setitem__(self, item_key:str, value:Any) -> None:
        self.redis.hset(self.key, item_key, value)
        
    def __delitem__(self, item_key: Any) -> None:
        self.redis.hdel(self.key, item_key)
        
    def __iter__(self) -> Iterator:
        return self.redis.hscan_iter(self.key)
        
    def __len__(self) -> int:
        return self.redis.hlen(self.key)
    
    def __contains__(self, key: object) -> bool:
        return self.redis.hexists(self.key, key)