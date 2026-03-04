"""
Topological Memory Map — Visual graph for indoor navigation.

Builds a graph of visited locations during exploration:
  - Nodes = keyframe images with visual features (HSV histogram + spatial grid)
  - Edges = traversable connections between adjacent locations (bidirectional)
  - Enables: backtracking, loop closure, shortest-path planning (A*)

This is critical for indoor competition where:
  - The robot may need to backtrack when it reaches dead ends
  - Previously seen corridors can guide navigation to new goals
  - Loop closure detects when the robot returns to a known location

Architecture:
  - Online graph construction: add nodes every N seconds or on significant
    view change (HSV histogram difference)
  - Edge weights: traversal cost (time-based); reverse edges 1.2x penalty
  - Visual loop closure: detect revisits via feature similarity (≥ 0.85
    cosine similarity, ≥ 5 node gap)
  - A* shortest path through the topo graph for global planning
  - Frontier detection: find dead-end / leaf nodes for exploration

Design inspired by:
  - Mobility VLA (Chiang et al., arXiv:2407.07775): hierarchical VLM +
    topological graph navigation — validates our architecture choice
  - Neural Topological SLAM (Chaplot et al., NeurIPS 2020): explicit
    topological maps for visual navigation
  - SLING (Chane-Sane et al., CoRL 2022): graph-based goal-conditioned
    navigation with learned subgoal proposals

References:
  [1] Chiang et al., \"Mobility VLA\", arXiv:2407.07775, 2024.
  [2] Chaplot et al., \"Neural Topological SLAM for Visual Navigation\",
      NeurIPS 2020.
  [3] Chane-Sane et al., \"Goal-Conditioned RL with Imagined Subgoals\",
      CoRL 2022.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TopoNode:
    """A node in the topological map."""
    node_id: int
    timestamp: float
    image: np.ndarray                          # keyframe (BGR)
    feature: Optional[np.ndarray] = None       # visual feature vector
    orientation: float = 0.0                   # IMU heading at this node
    position_estimate: Optional[Tuple[float, float]] = None  # estimated (x, y)
    visit_count: int = 1
    last_visit: float = 0.0

    def __hash__(self):
        return self.node_id

    def __eq__(self, other):
        return isinstance(other, TopoNode) and self.node_id == other.node_id


@dataclass
class TopoEdge:
    """An edge connecting two nodes."""
    from_id: int
    to_id: int
    cost: float = 1.0                # traversal cost (normalized time)
    traversal_count: int = 0
    last_traversal: float = 0.0
    action_sequence: List[Tuple[float, float]] = field(default_factory=list)  # (linear, angular) commands


@dataclass
class TopoMapConfig:
    """Configuration for the topological memory."""
    # Node creation
    min_node_distance: float = 2.0           # min seconds between new nodes
    scene_change_threshold: float = 0.25     # visual difference to force new node
    max_nodes: int = 500                     # maximum graph size

    # Loop closure
    loop_closure_threshold: float = 0.85     # feature similarity for loop closure
    loop_closure_min_gap: int = 5            # minimum node gap for loop closure

    # Feature extraction
    feature_method: str = "histogram"        # "histogram" (fast), "dinov2" (accurate)
    feature_dim: int = 256                   # histogram bins

    # Planning
    max_path_length: int = 50               # max A* path length


class TopologicalMemory:
    """
    Visual topological map for indoor navigation.

    Builds an online graph of visited locations and supports:
      - Backtracking: retrace path to a previous location
      - Loop closure: detect when returning to a known place
      - Global planning: A* through the graph to reach goals
      - Frontier detection: identify unexplored branches
    """

    def __init__(self, cfg: Optional[TopoMapConfig] = None):
        self.cfg = cfg or TopoMapConfig()
        self.nodes: Dict[int, TopoNode] = {}
        self.edges: Dict[int, Dict[int, TopoEdge]] = defaultdict(dict)  # adj list
        self._next_id: int = 0
        self._current_node: Optional[int] = None
        self._last_node_time: float = 0.0
        self._last_feature: Optional[np.ndarray] = None
        self._path_stack: List[int] = []  # for backtracking
        self._feature_extractor = None

    @property
    def current_node_id(self) -> Optional[int]:
        return self._current_node

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(e) for e in self.edges.values())

    def _extract_feature(self, image: np.ndarray) -> np.ndarray:
        """Extract visual feature from an image (fast histogram method)."""
        if self.cfg.feature_method == "histogram":
            # Multi-channel color histogram — fast and effective for scene change
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Also add spatial layout info (4x4 grid means)
            h, w = image.shape[:2]
            grid_feat = []
            for gy in range(4):
                for gx in range(4):
                    cell = image[
                        gy * h // 4 : (gy + 1) * h // 4,
                        gx * w // 4 : (gx + 1) * w // 4,
                    ]
                    grid_feat.extend(cell.mean(axis=(0, 1)).tolist())

            feat = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten(),
                np.array(grid_feat),
            ])
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat
        else:
            # Placeholder for DINOv2 features (shared with GoalMatcher)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            feat = hist.flatten()
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        return float(np.dot(feat1, feat2))

    def update(
        self,
        image: np.ndarray,
        orientation: float = 0.0,
        force_new_node: bool = False,
    ) -> Optional[int]:
        """
        Update the topological map with a new observation.

        Returns the node ID if a new node was created, None otherwise.
        """
        now = time.time()
        feature = self._extract_feature(image)

        # Check if we should create a new node
        create_node = force_new_node
        if not create_node:
            time_elapsed = now - self._last_node_time
            if time_elapsed >= self.cfg.min_node_distance:
                create_node = True

            # Scene change detection
            if self._last_feature is not None and not create_node:
                sim = self._compute_similarity(feature, self._last_feature)
                if sim < (1.0 - self.cfg.scene_change_threshold):
                    create_node = True

        if not create_node:
            self._last_feature = feature
            return None

        # Enforce max nodes
        if len(self.nodes) >= self.cfg.max_nodes:
            self._prune_oldest()

        # Create new node
        node_id = self._next_id
        self._next_id += 1

        node = TopoNode(
            node_id=node_id,
            timestamp=now,
            image=image.copy(),
            feature=feature,
            orientation=orientation,
            visit_count=1,
            last_visit=now,
        )
        self.nodes[node_id] = node

        # Connect to previous node
        if self._current_node is not None:
            prev_id = self._current_node
            cost = now - self._last_node_time
            edge = TopoEdge(
                from_id=prev_id,
                to_id=node_id,
                cost=max(0.1, cost),
                traversal_count=1,
                last_traversal=now,
            )
            self.edges[prev_id][node_id] = edge
            # Bidirectional (can reverse)
            rev_edge = TopoEdge(
                from_id=node_id,
                to_id=prev_id,
                cost=max(0.1, cost * 1.2),  # reversing is slightly more expensive
                traversal_count=0,
                last_traversal=now,
            )
            self.edges[node_id][prev_id] = rev_edge

        # Check for loop closure
        self._check_loop_closure(node_id, feature)

        # Update state
        self._current_node = node_id
        self._last_node_time = now
        self._last_feature = feature
        self._path_stack.append(node_id)

        logger.debug(
            "TopoMap: node %d created (total: %d nodes, %d edges)",
            node_id, len(self.nodes), self.num_edges,
        )

        return node_id

    def _check_loop_closure(self, new_id: int, new_feat: np.ndarray):
        """Detect loop closures: current observation matches a distant past node."""
        for node_id, node in self.nodes.items():
            if node_id == new_id:
                continue
            # Skip recent nodes (too close in the graph)
            if abs(new_id - node_id) < self.cfg.loop_closure_min_gap:
                continue
            # Already connected?
            if node_id in self.edges.get(new_id, {}):
                continue

            if node.feature is None:
                continue

            sim = self._compute_similarity(new_feat, node.feature)
            if sim >= self.cfg.loop_closure_threshold:
                # Loop closure detected!
                edge = TopoEdge(
                    from_id=new_id,
                    to_id=node_id,
                    cost=0.5,  # loop closure = shortcut
                    traversal_count=0,
                    last_traversal=time.time(),
                )
                self.edges[new_id][node_id] = edge
                self.edges[node_id][new_id] = TopoEdge(
                    from_id=node_id, to_id=new_id, cost=0.5,
                )

                node.visit_count += 1
                node.last_visit = time.time()

                logger.info(
                    "LOOP CLOSURE: node %d ↔ node %d (similarity=%.3f)",
                    new_id, node_id, sim,
                )

    def find_most_similar_node(self, image: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Find the node most visually similar to the given image.

        Returns (node_id, similarity) or None if no nodes exist.
        """
        if not self.nodes:
            return None

        feature = self._extract_feature(image)
        best_id = -1
        best_sim = -1.0

        for node_id, node in self.nodes.items():
            if node.feature is None:
                continue
            sim = self._compute_similarity(feature, node.feature)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

        if best_id >= 0:
            return best_id, best_sim
        return None

    def plan_path(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """
        A* shortest path from start to goal through the topological graph.

        Returns an ordered list of node IDs, or None if no path exists.
        """
        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        if start_id == goal_id:
            return [start_id]

        # A* with cost = edge cost
        open_set: List[Tuple[float, int]] = [(0.0, start_id)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start_id: 0.0}

        while open_set:
            _, current = heappop(open_set)

            if current == goal_id:
                # Reconstruct path
                path = [goal_id]
                while path[-1] != start_id:
                    path.append(came_from[path[-1]])
                return list(reversed(path))

            if len(came_from) > self.cfg.max_path_length:
                break  # search too deep

            for neighbor_id, edge in self.edges.get(current, {}).items():
                tentative = g_score[current] + edge.cost
                if tentative < g_score.get(neighbor_id, float("inf")):
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative
                    # Heuristic: 0 (Dijkstra) since we don't have spatial coords
                    heappush(open_set, (tentative, neighbor_id))

        return None  # no path found

    def get_backtrack_path(self, n_steps: int = 5) -> List[int]:
        """
        Get a path to backtrack N steps through the recent history.

        Useful for escaping dead ends.
        """
        if len(self._path_stack) <= 1:
            return []

        # Return the last N node IDs in reverse order
        backtrack = list(reversed(self._path_stack[-n_steps:]))
        return backtrack

    def get_frontier_nodes(self) -> List[int]:
        """
        Find frontier nodes: locations with few outgoing edges (unexplored directions).

        These are good candidates for exploration.
        """
        frontiers = []
        for node_id in self.nodes:
            n_edges = len(self.edges.get(node_id, {}))
            if n_edges <= 1:  # dead end or leaf node
                frontiers.append(node_id)
        return frontiers

    def _prune_oldest(self):
        """Remove the oldest, least-visited node to stay under max_nodes."""
        if not self.nodes:
            return

        # Score: lower = more prunable (old, rarely visited)
        worst_id = min(
            self.nodes.keys(),
            key=lambda nid: self.nodes[nid].visit_count * 10 + self.nodes[nid].last_visit,
        )

        # Don't prune the current node
        if worst_id == self._current_node:
            return

        # Remove node and all its edges
        del self.nodes[worst_id]
        if worst_id in self.edges:
            del self.edges[worst_id]
        for adj in self.edges.values():
            adj.pop(worst_id, None)
        if worst_id in self._path_stack:
            self._path_stack = [n for n in self._path_stack if n != worst_id]

    def status_str(self) -> str:
        loops = sum(
            1 for nid, edges in self.edges.items()
            for eid in edges
            if abs(nid - eid) >= self.cfg.loop_closure_min_gap
        )
        return (
            f"TopoMap: {len(self.nodes)} nodes, {self.num_edges} edges, "
            f"{loops} loop closures, current={self._current_node}"
        )

    def reset(self):
        """Clear the entire map."""
        self.nodes.clear()
        self.edges.clear()
        self._next_id = 0
        self._current_node = None
        self._last_node_time = 0.0
        self._last_feature = None
        self._path_stack.clear()
