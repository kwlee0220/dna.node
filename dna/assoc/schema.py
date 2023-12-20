from __future__ import annotations

from typing import Optional
from collections.abc import Generator, Iterable
import logging

from dna import NodeId
from .association import Association


class NodeAssociationClosure:
    __slots__ = ('nodes', '__pairs')
    
    def __init__(self, nodes:list[NodeId], pairs:list[tuple[NodeId,NodeId]]) -> None:
        self.nodes = sorted(nodes)
        self.__pairs = pairs
        
    def index(self, node:NodeId):
        return self.nodes.index(node)
        
    def pairs(self, node:Optional[NodeId]=None) -> list[tuple[NodeId,NodeId]]:
        if node:
            return [pair for pair in self.__pairs if node in pair]
        else:
            return self.__pairs
        
    @staticmethod
    def build_closures(pairs:list[tuple[NodeId,NodeId]]) -> list[NodeAssociationClosure]:
        def to_closure(nodes:Iterable[NodeId]) -> NodeAssociationClosure:
            sub_pairs = [pair for pair in pairs if (pair[0] in nodes)]
            return NodeAssociationClosure(nodes, sub_pairs)
            
        import networkx as nx
        graph = nx.Graph(pairs)
        return [to_closure(comp) for comp in list(nx.connected_components(graph))]
        
    def __iter__(self) -> Generator[NodeId, None, None]:
        return iter(self.nodes)
    
    def __bool__(self) -> bool:
        return len(self.nodes)
    
    def __contains__(self, v:NodeId|tuple[NodeId,NodeId]) -> bool:
        if isinstance(v, str):
            return v in self.nodes
        else:
            if v[0] > v[1]:
                v = (v[1], v[0])
            return v in self.__pairs
            
    def __eq__(self, other:object):
        if isinstance(other, NodeAssociationClosure):
            return self.__pairs == other.__pairs
        else:
            return False
            
    def __repr__(self) -> str:
        return f"{self.__pairs}"


class NodeAssociationSchema:
    __slots__ = ('nodes', '__pairs', 'closures', 'closure_map')
    
    def __init__(self, node_pairs:list[tuple[NodeId,NodeId]]) -> None:
        self.__pairs = node_pairs
        self.closures = NodeAssociationClosure.build_closures(node_pairs)
        
        self.nodes:list[NodeId] = []
        self.closure_map:dict[NodeId, NodeAssociationClosure] = dict()
        for closure in self.closures:
            for n in closure.nodes:
                self.closure_map[n] = closure
            self.nodes.extend(closure.nodes)
        self.nodes = sorted(self.nodes)
        
    def closure(self, node:NodeId) -> Optional[NodeAssociationClosure]:
        return self.closure_map.get(node)
    
    def pairs(self, *, node:Optional[NodeId]=None) -> list[tuple[NodeId,NodeId]]:
        if node:
            return [pair for pair in self.__pairs if node in pair]
        else:
            return self.__pairs
        
    def peers(self, node:NodeId) -> list[NodeId]:
        return [pair[1] if pair[0] == node else pair[0] for pair in self.__pairs if node in pair]
