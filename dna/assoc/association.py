from __future__ import annotations

from typing import Union, Optional
from collections.abc import Iterator, Iterable
from abc import ABCMeta, abstractmethod

from dna import NodeId, TrackId, TrackletId
from dna.support import iterables


def _GET_NODE(tracklet:TrackletId):
    return tracklet.node_id

class Association(metaclass=ABCMeta):
    @property
    @abstractmethod
    def tracklets(self) -> set[TrackletId]:
        """Association에 포함된 모든 tracklet의 식별자를 반환한다.

        Returns:
            set[TrackletId]: TrackletId set.
        """
        pass
    
    @property
    def nodes(self) -> set[NodeId]:
        """Association에 포함된 모든 tracklet의 node 식별자들을 반환한다.

        Returns:
            set[NodeId]: NodeId set.
        """
        return {trk.node_id for trk in self.tracklets}
    
    @property
    @abstractmethod
    def score(self) -> float:
        """본 association 정보의 confidence 점수를 반환한다.

        Returns:
            float: confidence 점수 (0~1)
        """
        pass
    
    @property
    @abstractmethod
    def ts(self) -> int:
        """본 association이 생성된 timestamp를 반환한다.

        Returns:
            int: timestamp in milli-seconds.
        """
        pass
        
    @abstractmethod
    def is_closed(self, *, node:Optional[NodeId]=None, tracklet:Optional[TrackletId]=None) -> bool:
        """본 association의 추후 변경 여부를 반환한다.
        'node' 인자가 지정된 경우에는 이 식별자의 node의 tracklet의 종료 여부를 반환하고,
        'tracklet'인자가 지정된 경우에는 지정된 식별자의 tracklet의 종료 여부를 반환한다.
        만일 'node', 'tracklet' 이 모두 지정되지 않은 경우에는 association에 포함된 모든 tracklet의 종료 여부를 반환한다.

        Args:
            node (Optional[NodeId], optional): 대상 node의 식별자. Defaults to None.
            tracklet (Optional[TrackletId], optional): 대상 tracklet의 식별자. Defaults to None.

        Returns:
            bool: 종료 여부.
        """
        pass
    
    @abstractmethod
    def copy(self) -> Association:
        pass
        
    def tracklet(self, node:NodeId) -> Optional[TrackletId]:
        """주어진 node 식별자에 해당하는 tracklet 식별자를 반환한다.
        만일 node 식별자에 해당하는 tracklet이 존재하지 않는 경우에는 None를 반환한다.

        Args:
            node_id (NodeId): node 식별자

        Returns:
            Optional[TrackletId]: 주어진 node 식별자에 해당하는 tracklet의 식별자. 해당 tracklet이 존재하지 않으면 None.
        """
        return iterables.find(self.tracklets, node, keyer=_GET_NODE)
        
    def track(self, node:NodeId) -> Optional[TrackId]:
        """주어진 node에 해당하는 track 식별자를 반환한다.
        만일 node에 해당하는 track이 존재하지 않는 경우에는 None이 반환된다.

        Args:
            node (NodeId): 검색 node 식별자.

        Returns:
            Optional[TrackId]: track 식별자.  node에 해당하는 track이 존재하지 않는 경우에는 None.
        """
        trk = self.tracklet(node)
        return trk.track_id if trk else None
    
    def is_subset(self, other:Union[Association,Iterable[TrackletId]], *, exclude_same=False) -> bool:
        other_trks = set(other.tracklets if isinstance(other, Association) else other)
        return self.tracklets.issubset(other_trks) and (not exclude_same or len(self.tracklets) != len(other_trks))
    
    def is_superset(self, other:Union[Association,Iterable[TrackletId]], *, exclude_same=False) -> bool:
        other_trks = set(other.tracklets if isinstance(other, Association) else other)
        return self.tracklets.issuperset(other_trks) and (not exclude_same or len(self.tracklets) != len(other_trks))
    
    def is_disjoint(self, other:Association) -> bool:
        """두 association이 서로 disjoint 여부를 반환한다.
        두 association 사이의 동일 tracklet를 포함하지 않는 경우를 disjoint라고 정의함.

        Args:
            other (Association): 비교 대상 association.

        Returns:
            bool: Disjoint인 경우는 True 그렇지 않은 경우는 False
        """
        return self.tracklets.isdisjoint(other.tracklets)
        
    def is_conflict(self, other:Association) -> bool:
        """두 association이 서로 conflict 여부를 반환한다.
        두 association 사이에 최소 1개 이상의 동일 node의 track이 존재하고,
        이들 중 최소 1개 이상의 node의 track 식별자가 다른 경우 conflict함.

        Args:
            other (Association): 비교 대상 association.

        Returns:
            bool: Conflict인 경우는 True 그렇지 않은 경우는 False
        """
        for trk in self.tracklets:
            track = other.track(trk.node_id)
            if track and trk.track_id != track:
                return True
        return False
        # overlap_nodes = self.intersect_nodes(other.nodes)
        # if overlap_nodes:
        #     return any(self.track(node) != other.track(node) for node in overlap_nodes)
        # else:
        #     return False
    
    def is_compatible(self, other:Association) -> bool:
        """두 association이 서로 compatible 여부를 반환한다.
        두 association 사이에 최소 1개 이상의 동일 node의 track이 존재하고,
        모든 동일 node의 track 식별자가 동일한 경우 compatible함.

        Args:
            other (Association): 비교 대상 association.

        Returns:
            bool: Compatible한 경우는 True 그렇지 않은 경우는 False
        """
        for trk in self.tracklets:
            track = other.track(trk.node_id)
            if track and trk.track_id != track:
                return False
        return True
        # overlap_nodes = self.intersect_nodes(other.nodes)
        # if overlap_nodes:
        #     return all(self.track(node) == other.track(node) for node in overlap_nodes)
        # else:
        #     return False
        
    def is_more_specific(self, assoc:Association) -> bool:
        # 본 association을 구성하는 tracklet의 수가 'assoc'의 tracklet 수보다 작다면
        # 'more-specific'일 수 없기 때문에 'Fase'를 반환한다.
        if len(self) < len(assoc):
            return False
        
        if self.is_superset(assoc):
            if len(self) > len(assoc):
                return True
            else:
                # self와 assoc은 서로 동일한 tracklet으로 구성된 closure인 경우
                return self.score > assoc.score
        else:
            return False
        
    def is_superior(self, other:Association) -> bool:
        if self.is_conflict(other):
            return self.score > other.score
        else:
            return self.is_more_specific(other)
    
    def intersect_nodes(self, other:Union[Association,Iterable[NodeId]]) -> list[NodeId]:
        other_nodes = other.nodes if isinstance(other, Association) else set(other)
        return self.nodes.intersection(other_nodes)
    
    def intersect_tracklets(self, other:Union[Association,Iterable[TrackletId]]) -> list[TrackletId]:
        other_tracklets = other.tracklets if isinstance(other, Association) else set(other)
        return self.tracklets.intersection(other_tracklets)
    
    def __len__(self) -> int:
        return len(self.tracklets)
    
    def __contains__(self, key:Union[TrackletId,NodeId,Iterable[Union[TrackletId,NodeId]]]) -> bool:
        if isinstance(key, TrackletId):
            return bool(iterables.find(self.tracklets, key))
        elif isinstance(key, str):  # NodeId
            return bool(iterables.find(self.tracklets, key, keyer=_GET_NODE))
        elif isinstance(key, Iterable):
            return all(subkey in self for subkey in key)
        else:
            raise ValueError(f'invalid key: {key}')
    
    def __iter__(self) -> Iterator[TrackletId]:
        return iter(self.tracklets)
    
    def __getitem__(self, index:NodeId) -> TrackletId:
        if trk := self.tracklet(index):
            return trk
        else:
            raise KeyError(f"invalid node index: {index}")
    
    def __eq__(self, other:object) -> bool:
        if isinstance(other, Association):
            return self.tracklets == other.tracklets
        else:
            return False
    
    def __repr__(self) -> str:
        trks_str = '-'.join([str(trk) for trk in sorted(self.tracklets)])
        closeds_str = ''.join(['X' if self.is_closed(tracklet=trk) else 'O' for trk in sorted(self.tracklets)])
        return f"{trks_str}: {self.score:.3f} ({closeds_str})"
        # return f"{trks_str}: {self.score:.3f}"


class BinaryAssociation(Association):
    __slots__ = ('_tracklets', '_score', '_ts', '_closeds')
    
    def __init__(self, tracklet1:TrackletId, tracklet2:TrackletId, score:float, ts:int) -> None:
        super().__init__()
        self._tracklets = {tracklet1, tracklet2}
        self._score = score
        self._ts = ts
        self._closeds:set[NodeId] = set()
    
    @property
    def tracklets(self) -> set[TrackletId]:
        return self._tracklets
    
    @property
    def score(self) -> float:
        return self._score
    
    @property
    def ts(self) -> int:
        return self._ts
        
    def is_closed(self, *, node:Optional[NodeId]=None, tracklet:Optional[TrackletId]=None) -> bool:
        if tracklet:
            node = tracklet.node_id
        if node:
            return node in self._closeds
        else:
            return len(self.tracklets) == len(self._closeds)
    
    def close(self, *, node:Optional[NodeId]=None, tracklet:Optional[TrackletId]=None) -> bool:
        if not node and not tracklet:
            raise ValueError(f'target node|tarcklet is not specificed')
        
        if tracklet:
            node = tracklet.node_id
        if node in self:
            self._closeds.add(node)
        else:
            raise KeyError(f'invalid key={node}')
        
    def copy(self) -> Association:
        assoc = BinaryAssociation(*self._tracklets, self.score, self.ts)
        assoc._closeds = self._closeds.copy()
        return assoc
        
    @staticmethod
    def from_row(args) -> BinaryAssociation:
        tracklet1 = TrackletId(args[0], args[1])
        tracklet2 = TrackletId(args[2], args[3])
        return BinaryAssociation(tracklet1, tracklet2, score=args[4], ts=args[5])
    
    def to_row(self) -> tuple[str,str,str,str,float,int]:
        trk1, trk2 = tuple(sorted(self._tracklets))
        return (trk1.node_id, trk1.track_id,
                trk2.node_id, trk2.track_id,
                self.score, self.ts)