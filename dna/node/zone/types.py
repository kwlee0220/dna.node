from __future__ import annotations

from typing import Optional
from enum import Enum
from dataclasses import dataclass, field


class ZoneRelation(Enum):
    Unassigned = 0
    Entered = 1
    Left = 2
    Inside = 3
    Through = 4
    Deleted = 5
    
    @staticmethod
    def parseRelationStr(rel_str:str) -> tuple[ZoneRelation, str]:
        def parseZoneId(expr:str) -> str:
            return expr[2:-1]
            
        if rel_str[0] == 'U':
            return ZoneRelation.Unassigned, None
        elif rel_str[0] == 'E':
            return ZoneRelation.Entered, parseZoneId(rel_str)
        elif rel_str[0] == 'L':
            return ZoneRelation.Left, parseZoneId(rel_str)
        elif rel_str[0] == 'I':
            return ZoneRelation.Inside, parseZoneId(rel_str)
        elif rel_str[0] == 'T':
            return ZoneRelation.Through, parseZoneId(rel_str)
        elif rel_str[0] == 'D':
            return ZoneRelation.Deleted, parseZoneId(rel_str)


@dataclass(frozen=True, eq=True, repr=False)
class ZoneExpression:
    relation: ZoneRelation
    zone_id: Optional[str] = field(default=None)
    
    @staticmethod
    def UNASSIGNED() -> ZoneExpression:
        return ZoneExpression(relation=ZoneRelation.Unassigned)
    @staticmethod
    def ENTERED(zone_id:str) -> ZoneExpression:
        return ZoneExpression(relation=ZoneRelation.Entered, zone_id=zone_id)
    @staticmethod
    def LEFT(zone_id:str) -> ZoneExpression:
        return ZoneExpression(relation=ZoneRelation.Left, zone_id=zone_id)

    def is_unassigned(self) -> bool:
        return self.relation == ZoneRelation.Unassigned
    def is_entered(self) -> bool:
        return self.relation == ZoneRelation.Entered
    def is_left(self) -> bool:
        return self.relation == ZoneRelation.Left
    def is_inside(self) -> bool:
        return self.relation == ZoneRelation.Inside
    def is_through(self) -> bool:
        return self.relation == ZoneRelation.Through
    def is_deleted(self) -> bool:
        return self.relation == ZoneRelation.Deleted
    
    @classmethod
    def parse_str(cls, expr_str:str) -> ZoneExpression:
        if expr_str is None:
            return None
        
        def parseZoneId(expr:str) -> str:
            return expr[2:-1]
        
        rel_str = expr_str[0]
        if rel_str == 'U':
            return ZoneExpression(relation=ZoneRelation.Unassigned)
        elif rel_str == 'E':
            return ZoneExpression(relation=ZoneRelation.Entered, zone_id=parseZoneId(expr_str))
        elif rel_str == 'L':
            return ZoneExpression(relation=ZoneRelation.Left, zone_id=parseZoneId(expr_str))
        elif rel_str == 'I':
            return ZoneExpression(relation=ZoneRelation.Inside, zone_id=parseZoneId(expr_str))
        elif rel_str == 'T':
            return ZoneExpression(relation=ZoneRelation.Through, zone_id=parseZoneId(expr_str))
        elif rel_str == 'D':
            return ZoneExpression(relation=ZoneRelation.Deleted)
    
    def __repr__(self) -> str:
        if self.relation == ZoneRelation.Unassigned:
            return f'U'
        elif self.relation == ZoneRelation.Entered:
            return f'E({self.zone_id})'
        elif self.relation == ZoneRelation.Left:
            return f'L({self.zone_id})'
        elif self.relation == ZoneRelation.Through:
            return f'T({self.zone_id})'
        elif self.relation == ZoneRelation.Inside:
            return f'I({self.zone_id})'
        elif self.relation == ZoneRelation.Deleted:
            return 'D'
        else:
            raise ValueError(f'invalid zone_relation: {self.relation}')
UNASSIGNED = ZoneExpression(relation=ZoneRelation.Unassigned)