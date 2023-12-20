from __future__ import annotations

from dataclasses import dataclass, field
from omegaconf import OmegaConf

from dna import Point, Box
from dna.support import iterables


def from_sql_point(pt_str:str) -> Point:
    nums = pt_str.replace('(', '').replace(')', '').split(',')
    return Point([float(n) for n in nums])

def to_sql_point(pt:Point) -> str:
    return f'({pt.x},{pt.y})'

def from_sql_box(pt_str:str) -> Box:
    nums = [float(num) for num in pt_str.replace('(', '').replace(')', '').split(',')]
    br, tl = tuple(Point(xy) for xy in iterables.buffer_iterable(nums, 2))
    return Box.from_points(tl, br)

def to_sql_box(box:Box) -> str:
    pt1 = to_sql_point(Point(box.tl))
    pt2 = to_sql_point(Point(box.br))
    return ','.join([pt1, pt2])


_DEFAULT_POSTGRES_HOST = 'localhost'
_DEFAULT_POSTGRES_PORT = 5432
_DEFAULT_POSTGRES_USER = 'dna'
_DEFAULT_POSTGRES_PASSWORD = 'urc2004'
_DEFAULT_POSTGRES_DBNAME = '/dna'
_DB_CONF_KEYS = {'db_host', 'db_port', 'db_dbname', 'db_user', 'db_password'}

@dataclass(frozen=True)
class SQLConnector:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    
    @classmethod
    def from_url(cls, url:str) -> SQLConnector:
        import urllib
        
        result = urllib.parse.urlparse(url)
        if result.scheme != 'postgresql':
            import sys
            raise ValueError(f"invalid PostgreSQL URL: {url}")
        
        dbname = result.path if result.path else _DEFAULT_POSTGRES_DBNAME
        dbname = dbname[1:]
        return cls(host=result.hostname if result.hostname else _DEFAULT_POSTGRES_HOST,
                   port=result.port if result.port else _DEFAULT_POSTGRES_PORT,
                   user=result.username if result.username else _DEFAULT_POSTGRES_USER,
                   password=result.password if result.password else _DEFAULT_POSTGRES_PASSWORD,
                   dbname=dbname)
    
    def connect(self):
        import psycopg2
        from dataclasses import asdict
        return psycopg2.connect(**asdict(self))