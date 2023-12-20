from __future__ import annotations

from typing import Union, Optional
from collections.abc import Generator
from collections import defaultdict
import logging

import numpy as np

from dna import NodeId, TrackletId
from dna.event import EventProcessor, TrackDeleted, TrackFeature
from dna.node.zone import ZoneRelation
from dna.event.tracklet_store import TrackletStore
from dna.track.utils import cosine_distance
from .association import Association, BinaryAssociation


class IncomingLink:
    __slots__ = ('node_id', 'enter_zone', 'exit_zone', 'transition_time', 'node_travel_time')
    
    def __init__(self, from_node_id:str, enter_zone:str, exit_zone:str,
                 transition_time:float, node_travel_time:float) -> None:
        self.node_id = from_node_id     # 진입 node의 식별자.
        self.enter_zone = enter_zone    # 
        self.exit_zone = exit_zone
        self.transition_time = transition_time
        self.node_travel_time = node_travel_time
        
    @property
    def transition_ms(self):
        return round(self.transition_time * 1000)
        
    @property
    def node_travel_ms(self):
        return round(self.node_travel_time * 1000)
        
    def __eq__(self, other: object) -> bool:
        if other and isinstance(other, IncomingLink):
            return self.node_id == other.node_id  \
                    and self.enter_zone == other.enter_zone \
                    and self.exit_zone == other.exit_zone
        else:
            return False
        
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
        
    def __hash__(self) -> int:
        return hash(self.node_id, self.enter_zone, self.exit_zone)
    
    def __repr__(self) -> str:
        enter_zone_str = self.enter_zone if self.enter_zone else '*'
        exit_zone_str = self.exit_zone if self.exit_zone else '*'
        return ( f'{self.node_id}[{enter_zone_str}->{exit_zone_str}], '
                f'{self.node_travel_ms:.3f}, {self.transition_ms:.3f}' )
    

NODE_NETWORK:dict[str,dict[str,list[IncomingLink]]] = {
    'etri:04': {
        'A': [
                IncomingLink('etri:07', None, 'A', 0, 3),
            ],
        'B': [
            ],
        'C': [
            ],
    },
    'etri:05': {
        'A': [
                IncomingLink('etri:07', None, 'A', 0, 3),
            ], 
        'B': [
            ],
        'C': [
            ],
    },
    'etri:06': {
        'A': [
                IncomingLink('etri:07', None, 'A', 0, 3),
            ],
        'B': [
            ],
        'C': [
            ],
    },
    'etri:07': {
        'A': [
                IncomingLink('etri:04', None, 'A', 2, 3),
                IncomingLink('etri:05', None, 'A', 2, 3),
                IncomingLink('etri:06', None, 'A', 2, 3)
            ],
    }
}


class AssociationSession:
    __slots__ = ('id', 't_features', 'enter_zone', 'association', 'logger')
    
    def __init__(self, id:TrackletId, *, logger:Optional[logging.Logger]=None) -> None:
        self.id = id
        self.t_features:list[TrackFeature] = []
        self.enter_zone = None
        self.association:Association = None
        self.logger = logger
        
    def incoming_links(self):
        target_node_links = NODE_NETWORK.get(self.id.node_id, dict())
        return target_node_links.get(self.enter_zone, [])
        
    @property
    def ts(self) -> int:
        return self.t_features[-1].ts
        
    def append(self, t_feature:TrackFeature) -> None:
        self.t_features.append(t_feature)
        if not self.enter_zone:
            # 아직 enter-zone이 결정되지 않은 상태면, 새롭게 추가된 feature 정보에서
            # enter-zone 진입 여부를 확인한다.
            zone_rel, zone_id = ZoneRelation.parseRelationStr(t_feature.zone_relation)
            if zone_rel == ZoneRelation.Entered or zone_rel == ZoneRelation.Inside:
                self.enter_zone = zone_id
    
    def calc_topk_distance(self, track_features:list[TrackFeature], top_k:int, logger:logging.Logger) -> float:
        if track_features:
            peer_features = np.array([t_f.feature for t_f in track_features])
            distances = cosine_distance(self.features(), peer_features, True).min(axis=0)
            
            top_k_dists = np.partition(distances, top_k)[:top_k] if len(distances) > top_k else distances
            top_k_dist = max(top_k_dists)
            # if logger and logger.isEnabledFor(logging.DEBUG):
            #     sorted_dists = sorted(distances)
            #     raw_dist_str = ','.join([f"{dist:.3f}" for dist in sorted_dists])
            #     logger.debug((f"{self.id}<->{track_features[0].tracklet_id}: "
            #                   f"top_k_dist={top_k_dist:.3f}, raw_dists={raw_dist_str}"))
            return top_k_dist
        else:
            return 1
    
    def __len__(self) -> int:
        return len(self.t_features)
    
    def __iter__(self):
        return (t_feature for t_feature in self.t_features)
    
    def __getitem__(self, index) -> Union[TrackFeature, list[TrackFeature]]:
        return self.t_features[index]
                    
    def features(self) -> np.ndarray:
        return np.array([tf.feature for tf in self.t_features])
    
    def __repr__(self) -> str:
        peer = f', peer: {self.association.tracklet2} ({self.association.score:.3f})' if self.association else ''
        return f'{self.id}, enter={self.enter_zone}, n_features=[{len(self.t_features)}]{peer}'


class FeatureBasedTrackletAssociator(EventProcessor):
    __slots__ = ( 'store', 'listen_nodes', 'prefix_length', 'associations', 'top_k', 'logger' )
    
    def __init__(self,
                 store:TrackletStore,
                 listen_nodes:set[str],
                 *,
                 prefix_length:int=5,
                 top_k:int=4,
                 early_stop_score:float=1,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.store = store
        self.listen_nodes = listen_nodes
        self.prefix_length = prefix_length
        self.top_k = top_k
        self.early_stop_score = early_stop_score
        self.sessions:dict[TrackletId,AssociationSession] = dict()
        self.early_stoppeds:dict[TrackletId,list[NodeId]] = defaultdict(list)
        self.logger = logger
        
    def handle_event(self, ev:TrackFeature) -> None: 
        tracklet_id = ev.tracklet_id
        
        if ev.zone_relation == 'D':
            # track이 종료되면 더 이상 수행할 association이 없기 때문에,
            # 해당 traklet의 session을 활용한 마지막 association을 시도하고 session을 제거한다.
            session = self.sessions.pop(tracklet_id, None)
            if session and session.enter_zone:
                self.publish_associations(session)
            self._publish_event(TrackDeleted(node_id=ev.node_id, track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts))
            self.early_stoppeds.pop(tracklet_id, None)
        else:
            # Association이 이미 완료된 상태가 아닌 경우에 association 작업을 수행한다.
            session = self.get_or_create_session(self.sessions, tracklet_id)
            session.append(ev)
            
            # Association에 필요한 최소 feature 수가 넘으면 association을 수행한다.
            # Association을 위해서는 대상 track은 최소 'enter zone'이 결정된 상태이어야 한다.
            # 만일 그렇지 않다면 '최소 feature 수'가 넘은 상태여도 association을 진행할 수 없다.
            n_extends = len(session) - self.prefix_length
            if (n_extends >= 0 and (n_extends % 4) == 0) and session.enter_zone:
                self.publish_associations(session)
        
    def get_or_create_session(self, sessions, tracklet_id:TrackletId) -> AssociationSession:
        session = sessions.get(tracklet_id)
        if not session:
            session = AssociationSession(tracklet_id)
            sessions[tracklet_id] = session
        return session

    def publish_associations(self, session:AssociationSession) -> bool:
        target_node_links = NODE_NETWORK.get(session.id.node_id, dict())
        incoming_links = target_node_links.get(session.enter_zone, [])
        for incoming_link in incoming_links:
            if incoming_link.node_id not in self.early_stoppeds[session.id]:
                for assoc in self.associate(session, incoming_link):
                    self._publish_event(assoc)
            else:
                pass
            
    def associate(self, session:AssociationSession, incoming_link:IncomingLink) -> Generator[Association,None,None]:
        # association 후보 tracklet을 선정한다.
        candidate_tracklets = self._find_candidate_tracklets(session, incoming_link)
        
        # 후보 tracklet들과 대상으로 feature score를 계산한다.
        candidate_scores = [(trk_id, self.calc_match_score(session, trk_id)) for trk_id in candidate_tracklets]
        pass
        candidate_scores = sorted(candidate_scores, key=lambda t:t[1], reverse=True)
        if candidate_scores and candidate_scores[0][1] >= self.early_stop_score:
            self.early_stoppeds[session.id].append(candidate_scores[0][0].node_id)
        for trk, score in candidate_scores:
            yield BinaryAssociation(session.id, trk, score, session.ts)
                
    def _find_candidate_tracklets(self, session:AssociationSession, incoming_link:IncomingLink) -> list[TrackletId]:
        incoming_node = incoming_link.node_id
            
        # incoming link로부터 이전 node의 식별자와 비교 대상 시간 구간을 계산하고,
        # 해당 node에서의 구해진 시간 구간에 TrackEvent를 발생시킨 tracklet을 검색한다.
        end_ts = session[0].ts - incoming_link.transition_ms
        begin_ts = end_ts - incoming_link.node_travel_ms
        tracklets_by_time = set(self.store.list_tracklets_in_range(incoming_node, begin_ts, end_ts))
        
        # incoming node에 예상되는 시간 구간의 tracklet이 존재하지 않는다면
        # 해당 incoming node에서 발생된 feature들과 association을 시도할 여지가 없음.
        if not tracklets_by_time:
            # if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            #     self.logger.debug((f"fails to match to {session.id}: no peer tracklets "
            #                        f"between {begin_ts}-{end_ts} at {incoming_node} now."))
            return []
            
        # 주어진 enter_zone과 exit_zone을 거쳐간 incoming_node에서 발생된 tracklet motion들을 검색한다.
        tracklet_motions = self.store.list_tracklet_motions(incoming_node,
                                                            enter_zone=incoming_link.enter_zone,
                                                            exit_zone=incoming_link.exit_zone)
        tracklets_by_motion = {tracklet_motion.tracklet_id for tracklet_motion in tracklet_motions}
        if not tracklets_by_motion:
            # if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            #     enter_str = incoming_link.enter_zone if incoming_link.enter_zone else 'ANY'
            #     exit_str = incoming_link.exit_zone if incoming_link.exit_zone else 'ANY'
            #     self.logger.debug((f"fails to match to {session.id}: no peer tracklets through "
            #                         f"enter({enter_str}) and exit({exit_str}) at {incoming_node} now."))
            return []
            
        # incoming node에서 검출된 tracklet들 중에서 시간 조건과 exit_zone 조건을 고려하여 최종적으로
        # association 대상 tracklet을 찾는다
        return (tracklets_by_time & tracklets_by_motion)
        
    def calc_match_score(self, session:AssociationSession, peer_tracklet:TrackletId) -> float:
        # Match 대상 track에서 생성된 feature들을 TrackletStore에서 읽어서
        # 본 session을 통해 수집된 feature들과의 matching score를 계산한다.
        peer_features = self.store.read_tracklet_features(peer_tracklet)
        topk_dist = session.calc_topk_distance(peer_features, self.top_k, self.logger)
        return 1 - topk_dist
        