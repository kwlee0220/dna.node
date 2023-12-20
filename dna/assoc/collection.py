from __future__ import annotations

from typing import Union, Optional
from collections.abc import Iterable, Generator, Callable
import logging

from dna import TrackletId
from dna.event import EventProcessor
from dna.support import iterables
from .association import Association


class ExactMatch:
    def __init__(self, key:Union[Association,list[TrackletId]]):
        self.key = key.tracklets if isinstance(key, Association) else key
    def __call__(self, assoc:Association) -> bool:
        return self.key == assoc.tracklets

class PartialMatch:
    def __init__(self, key:Union[Association,Iterable[TrackletId],TrackletId]):
        if isinstance(key, Association):
            self.key = key.tracklets
        elif isinstance(key, Iterable):
            self.key = key
        elif isinstance(key, TrackletId):
            self.key = [key]
        else:
            raise ValueError(f"invalid key: {key}")
    def __call__(self, assoc:Association) -> bool:
        return self.key in assoc
    
class MoreSpecificMatch:
    def __init__(self, key:Association):
        self.key = key
    def __call__(self, assoc:Association) -> bool:
        return assoc.is_more_specific(self.key)
    
class LessSpecificMatch:
    def __init__(self, key:Association):
        self.key = key
    def __call__(self, assoc:Association) -> bool:
        return self.key.is_more_specific(assoc)


class AssociationCollection:
    def __init__(self, *,
                 keep_best_association_only:bool=False,
                 logger:Optional[logging.Logger]=None) -> None:
        self.collection:list[Association] = []
        self.keep_best_association_only = keep_best_association_only
        self.logger = logger
        
    def get(self, key:list[TrackletId]) -> Optional[Association]:
        """주어진 tracklet 들로 구성된 association을 검색한다.
        만일 해당 association이 없는 경우는 None을 반환한다.

        Args:
            key (list[TrackletId]): 검색에 사용할 tracklet list.

        Returns:
            Optional[Association]: 검색된 association 객체. 검색에 실패한 경우에는 None.
        """
        _, assoc = self.get_indexed(key)
        return assoc
        
    def get_indexed(self, key:list[TrackletId]) -> tuple[int,Optional[Association]]:
        for idx, assoc in enumerate(self.collection):
            if assoc.tracklets == key:
                return idx, assoc
        return -1, None
    
    def query(self, condition:Callable[[Association],bool],
              *,
              include_index:bool=False) -> Generator[Union[Association,tuple[int, Association]], None, None]:
        """주어진 condition을 만족하는 모든 association 객체들을 반환한다.

        Args:
            condition (Callable[[Association],bool]): 검색에 사용할 조건 객체.
            include_index (bool): 검색 결과에 검색된 association의 index 포함 여부.

        Returns:
             Generator[Union[Association,tuple[int, Association]]], None, None]: 조건을 만족하는 association을 반환하는 generator
        """
        for idx, assoc in enumerate(self.collection):
            if condition(assoc):
                if include_index:
                    yield idx, assoc
                else:
                    yield assoc
                
    def query_first(self, condition:Callable[[Association],bool],
                    *,
                    include_index=False) -> Association|tuple[int, Optional[Association]]:
        for idx, assoc in enumerate(self.collection):
            if condition(assoc):
                return ((idx, assoc) if include_index else assoc)
        return ((-1, None) if include_index else None)
    
    def add(self, assoc:Association) -> bool:
        if self.keep_best_association_only:
            return self._add_best_association(assoc)
        else:
            return self._guarded_add(assoc)
                
    def _add_best_association(self, assoc:Association) -> bool:
        # 새로 삽입할 association과 충돌을 발생시키는 기존 association들을 검색한다.
        candidates = [self.query_first(condition=PartialMatch(trk), include_index=True) for trk in assoc.tracklets]
        candidates = {cand[0]:cand[1] for cand in candidates if cand[1]}
        if not candidates:
            # 충돌하는 association이 없는 경우는 새 association을 삽입한다.
            return self._guarded_add(assoc)
        
        #
        # 충돌하는 association들이 존재하는 경우
        #
        
        # 새로 추가될 association보다 우월한 association을 검색한다.
        any_superior = iterables.first(candidate for _, candidate in candidates.items() if candidate.is_superior(assoc))
        if any_superior:
            # 우월한 association이 존재하면 새 association 삽입을 생략한다.
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'ignore: new={assoc}, existing={any_superior}')
            return False
        
        #
        # 기존 candidate들 보다 새로 추가할 association이 superior한 경우
        #
        
        # 기존 less-speicific한 association들은 모두 제거한다.
        less_specifics = [cand for cand in candidates.items() if assoc.is_more_specific(cand[1])]
        for less_idx, less_assoc in sorted(less_specifics, key=lambda item:item[0], reverse=True):
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f'remove: less-specific={less_assoc}, new={assoc}')
            self.collection.pop(less_idx)
            
        # 새로 추가될 association과 conflict한 모든 기존 association들에 대해
        # conflict를 유발하는 tracklet을 제거한 후 re-add를 시도한다.
        conflicts = [cand for cand in candidates.items() if assoc.is_conflict(cand[1])]
        
        # re-add 전에 새 association을 먼저 삽입한다.
        self._guarded_add(assoc)
        
        for conflict_idx, conflict_assoc in sorted(conflicts, key=lambda item:item[0], reverse=True):
            # conflict를 발생하는 association을 collection에서 제거
            prev_assoc = self.collection.pop(conflict_idx)
            
            # conflict를 발생하는 association에서 conflict를 유발하는 tracklet들을 모두 삭제한 association 생성.
            # print(f'assoc={assoc}, conflict_assoc={conflict_assoc}')
            for conflict_trk in assoc.intersect_tracklets(conflict_assoc):
                # 이전 iteration에서 특정 tracklet을 삭제할 때, side-effect로 다른 tracklet도 함께 지워질 수 있기 때문에 확인 필요.
                if conflict_trk in conflict_assoc:
                    conflict_assoc = conflict_assoc.remove_node(conflict_trk.node_id)
                # print(f"\tassoc={assoc}, {conflict_trk.node_id} -> conflict_assoc={conflict_assoc}")
                if conflict_assoc is None:
                    break
            
            # conflict 유발 tracklet을 제거한 association이 empty가 아니라면 collection에 재 삽입을 시도한다.
            done = conflict_assoc and self._guarded_add(conflict_assoc)
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                if done:
                    self.logger.info(f'resolve conflict: {prev_assoc} -> {conflict_assoc}')
                else:
                    self.logger.info(f'remove conflict: {prev_assoc}, new={assoc}')
        return True
    
    def _guarded_add(self, assoc:Association) -> bool:
        if len(self) == 0:
            self.collection.append(assoc)
            return True
        
        # collection에 이미 추가하려는 association보다 더 specific한 association이 하나라도 이미 존재하는 확인한다.
        prev = self.query_first(MoreSpecificMatch(assoc))
        if prev:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.info(f'ignore: {assoc}, existing={prev}')
            return False
        
        # 동일 tracklet으로 구성된 association이 이미 존재하는 경우를 확인한다.
        idx, prev = self.get_indexed(assoc.tracklets)
        if prev:
            # score 값을 기준으로 update 여부를 결정한다.
            if assoc.score > prev.score:
                self.collection[idx] = assoc
                return True
            else:
                return False
            
        # 추가하려는 association의 subset tracklet으로 구성된 association들을 모두 제거한다.
        less_specifics = self.query(LessSpecificMatch(assoc), include_index=True)
        less_specifics = sorted(less_specifics, key=lambda t:t[0], reverse=True)
        for idx, prev in less_specifics:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                victim = self.collection[idx]
                self.logger.info(f'remove: {victim}, new={assoc}')
            self.pop(idx)
            
        self.collection.append(assoc)
        return True
        
    def remove(self, key:list[TrackletId]) -> Association:
        idx, _ = self.get_indexed(key)
        if idx >= 0:
            return self.collection.pop(idx)
        else:
            return None

    def pop(self, index:int) -> Association:
        return self.collection.pop(index)
    
    def remove_cond(self, cond:Callable[[Association],bool]) -> list[Association]:
        length = len(self.collection)
        removeds = []
        for idx in range(length-1, -1, -1):
            if cond(self.collection[idx]):
                removeds.append(self.collection.pop(idx))
        return removeds
            
    def __len__(self) -> int:
        return len(self.collection)
            
    def __bool__(self) -> int:
        return bool(self.collection)
        
    def __iter__(self) -> Iterable[Association]:
        return iter(self.collection)
    
    def __getitem__(self, index:int) -> Association:
        return self.collection[index]
        
    def clear(self) -> None:
        self.collection.clear()


class AssociationCollector(EventProcessor):
    def __init__(self, 
                 *,
                 collection:Optional[AssociationCollection]=None,
                 publish_on_update:bool=False) -> None:
        super().__init__()
        
        self.collection = collection if collection is not None else AssociationCollection()
        self.publish_on_update = publish_on_update
    
    def handle_event(self, ev:Association) -> None:
        if isinstance(ev, Association):
            if self.collection.add(ev) and self.publish_on_update:
                self._publish_event(ev)
        
    def __iter__(self) -> Iterable[Association]:
        return iter(self.collection)
    
    def collect(self, assoc:Association) -> None:
        if self.keep_best_association_only:
            for trk in assoc.tracklets:
                prev_idx, prev_assoc = self.query_first(condition=PartialMatch(trk), include_index=True)
                if prev_assoc:
                    if assoc.score > prev_assoc.score or prev_assoc.is_subset(assoc, exclude_same=True):
                        self.collection.pop(prev_idx)
                        self.guarded_add(assoc)
                        if self.publish_on_update:
                            self._publish_event(assoc)
                    return
            self.guarded_add(assoc)
            if self.publish_on_update:
                self._publish_event(assoc)
        else:
            ok = self.guarded_add(assoc)
            if ok and self.publish_on_update:
                self._publish_event(assoc)