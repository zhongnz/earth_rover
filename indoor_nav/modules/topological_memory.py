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

import html
import json
import logging
import os
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
    exit_label: Optional[str] = None
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

    def _connect_nodes(
        self,
        from_id: int,
        to_id: int,
        cost: float,
        now: float,
        *,
        exit_label: Optional[str] = None,
    ):
        """Create or refresh a bidirectional traversability edge."""
        if from_id == to_id:
            return

        cost = max(0.1, float(cost))
        forward = self.edges[from_id].get(to_id)
        if forward is None:
            self.edges[from_id][to_id] = TopoEdge(
                from_id=from_id,
                to_id=to_id,
                cost=cost,
                traversal_count=1,
                last_traversal=now,
                exit_label=exit_label,
            )
        else:
            forward.cost = min(forward.cost, cost)
            forward.traversal_count += 1
            forward.last_traversal = now
            if exit_label is not None:
                forward.exit_label = exit_label

        reverse_cost = max(0.1, cost * 1.2)
        reverse = self.edges[to_id].get(from_id)
        if reverse is None:
            self.edges[to_id][from_id] = TopoEdge(
                from_id=to_id,
                to_id=from_id,
                cost=reverse_cost,
                traversal_count=0,
                last_traversal=now,
                exit_label="back",
            )
        else:
            reverse.cost = min(reverse.cost, reverse_cost)
            reverse.last_traversal = now

    def _match_existing_node(
        self,
        feature: np.ndarray,
        *,
        exclude_id: Optional[int] = None,
    ) -> Optional[Tuple[int, float]]:
        """Return the most similar existing node above the loop-closure threshold."""
        best_id = -1
        best_sim = -1.0
        for node_id, node in self.nodes.items():
            if node_id == exclude_id or node.feature is None:
                continue
            sim = self._compute_similarity(feature, node.feature)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id
        if best_id >= 0 and best_sim >= self.cfg.loop_closure_threshold:
            return best_id, best_sim
        return None

    def update(
        self,
        image: np.ndarray,
        orientation: float = 0.0,
        force_new_node: bool = False,
        exit_label: Optional[str] = None,
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

        if force_new_node and self.nodes:
            matched = self._match_existing_node(feature, exclude_id=self._current_node)
            if matched is not None:
                matched_id, matched_sim = matched
                prev_id = self._current_node
                if prev_id is not None:
                    cost = now - self._last_node_time
                    self._connect_nodes(prev_id, matched_id, cost, now, exit_label=exit_label)

                node = self.nodes[matched_id]
                node.visit_count += 1
                node.last_visit = now
                node.orientation = orientation
                node.image = image.copy()

                self._current_node = matched_id
                self._last_node_time = now
                self._last_feature = feature
                self._path_stack.append(matched_id)

                logger.info(
                    "TOPO RELOCALIZE: matched existing node %d (similarity=%.3f)",
                    matched_id,
                    matched_sim,
                )
                return matched_id

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
            self._connect_nodes(prev_id, node_id, cost, now, exit_label=exit_label)

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

    def get_exit_label(self, from_id: int, to_id: int) -> Optional[str]:
        edge = self.edges.get(from_id, {}).get(to_id)
        if edge is None:
            return None
        return edge.exit_label

    def plan_to_nearest_frontier(
        self,
        start_id: int,
        *,
        exclude_ids: Optional[Set[int]] = None,
    ) -> Optional[List[int]]:
        """Return the shortest path from start to the nearest frontier node."""
        if start_id not in self.nodes:
            return None
        exclude = exclude_ids or set()
        best_path: Optional[List[int]] = None
        best_cost = float("inf")
        for frontier_id in self.get_frontier_nodes():
            if frontier_id == start_id or frontier_id in exclude:
                continue
            path = self.plan_path(start_id, frontier_id)
            if not path or len(path) < 2:
                continue
            cost = 0.0
            for a, b in zip(path[:-1], path[1:]):
                edge = self.edges.get(a, {}).get(b)
                cost += edge.cost if edge is not None else 1.0
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

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

    def _loop_closure_count(self) -> int:
        return sum(
            1 for nid, edges in self.edges.items()
            for eid in edges
            if abs(nid - eid) >= self.cfg.loop_closure_min_gap
        )

    def to_debug_dict(self) -> dict:
        """Serialize the in-memory topo graph into a JSON-friendly structure."""
        node_records = []
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            outgoing = []
            for neighbor_id in sorted(self.edges.get(node_id, {})):
                edge = self.edges[node_id][neighbor_id]
                outgoing.append(
                    {
                        "to_id": int(neighbor_id),
                        "cost": float(edge.cost),
                        "traversal_count": int(edge.traversal_count),
                        "last_traversal": float(edge.last_traversal),
                        "exit_label": edge.exit_label,
                    }
                )

            node_records.append(
                {
                    "node_id": int(node_id),
                    "timestamp": float(node.timestamp),
                    "last_visit": float(node.last_visit),
                    "visit_count": int(node.visit_count),
                    "orientation": float(node.orientation),
                    "is_current": node_id == self._current_node,
                    "is_frontier": len(self.edges.get(node_id, {})) <= 1,
                    "thumbnail_path": f"nodes/node_{node_id:04d}.jpg",
                    "neighbors": outgoing,
                }
            )

        edge_records = []
        for from_id in sorted(self.edges):
            for to_id in sorted(self.edges[from_id]):
                edge = self.edges[from_id][to_id]
                edge_records.append(
                    {
                        "from_id": int(from_id),
                        "to_id": int(to_id),
                        "cost": float(edge.cost),
                        "traversal_count": int(edge.traversal_count),
                        "last_traversal": float(edge.last_traversal),
                        "exit_label": edge.exit_label,
                    }
                )

        return {
            "summary": {
                "num_nodes": int(len(self.nodes)),
                "num_edges": int(self.num_edges),
                "loop_closures": int(self._loop_closure_count()),
                "current_node_id": self._current_node,
                "frontier_nodes": self.get_frontier_nodes(),
                "path_stack": list(self._path_stack),
            },
            "nodes": node_records,
            "edges": edge_records,
        }

    def _build_debug_html(self, data: dict) -> str:
        """Render a lightweight HTML report for the topo graph."""
        summary = data["summary"]
        node_cards = []
        for node in data["nodes"]:
            neighbor_items = []
            for neighbor in node["neighbors"]:
                label = neighbor["exit_label"] or "unlabeled"
                neighbor_items.append(
                    "<li>"
                    f"{neighbor['to_id']} via {html.escape(label)} "
                    f"(cost={neighbor['cost']:.2f}, visits={neighbor['traversal_count']})"
                    "</li>"
                )
            frontier_badge = "<span class='badge frontier'>frontier</span>" if node["is_frontier"] else ""
            current_badge = "<span class='badge current'>current</span>" if node["is_current"] else ""
            node_cards.append(
                "<article class='card'>"
                f"<img src='{html.escape(node['thumbnail_path'])}' alt='node {node['node_id']}'>"
                "<div class='body'>"
                f"<h3>Node {node['node_id']} {current_badge} {frontier_badge}</h3>"
                f"<p>visits={node['visit_count']} orientation={node['orientation']:.1f}</p>"
                "<ul>"
                + "".join(neighbor_items or ["<li>No outgoing edges</li>"])
                + "</ul>"
                "</div></article>"
            )

        edge_rows = []
        for edge in data["edges"]:
            edge_rows.append(
                "<tr>"
                f"<td>{edge['from_id']}</td>"
                f"<td>{edge['to_id']}</td>"
                f"<td>{html.escape(edge['exit_label'] or 'unlabeled')}</td>"
                f"<td>{edge['cost']:.2f}</td>"
                f"<td>{edge['traversal_count']}</td>"
                "</tr>"
            )

        frontier_text = ", ".join(str(node_id) for node_id in summary["frontier_nodes"]) or "none"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Topo Debug</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f9; color: #111; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .metric {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 12px; overflow: hidden; }}
    .card img {{ width: 100%; height: 180px; object-fit: cover; background: #ddd; display: block; }}
    .body {{ padding: 12px; }}
    .body ul {{ padding-left: 18px; margin: 8px 0 0; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-left: 6px; }}
    .badge.current {{ background: #d8f5d0; }}
    .badge.frontier {{ background: #ffe7b8; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #ddd; border-radius: 12px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }}
    code {{ background: #ececf1; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Topological Map Debug</h1>
  <p>Current node: <code>{summary['current_node_id']}</code> | Frontiers: <code>{html.escape(frontier_text)}</code></p>
  <section class="summary">
    <div class="metric"><strong>Nodes</strong><div>{summary['num_nodes']}</div></div>
    <div class="metric"><strong>Edges</strong><div>{summary['num_edges']}</div></div>
    <div class="metric"><strong>Loop Closures</strong><div>{summary['loop_closures']}</div></div>
    <div class="metric"><strong>Path Stack</strong><div><code>{html.escape(str(summary['path_stack']))}</code></div></div>
  </section>
  <h2>Nodes</h2>
  <section class="grid">
    {''.join(node_cards)}
  </section>
  <h2>Edges</h2>
  <table>
    <thead><tr><th>From</th><th>To</th><th>Exit</th><th>Cost</th><th>Traversals</th></tr></thead>
    <tbody>{''.join(edge_rows)}</tbody>
  </table>
</body>
</html>
"""

    def export_debug_bundle(self, output_dir: str) -> dict:
        """Write a JSON snapshot, node thumbnails, and HTML debug page."""
        os.makedirs(output_dir, exist_ok=True)
        node_dir = os.path.join(output_dir, "nodes")
        os.makedirs(node_dir, exist_ok=True)

        for node_id, node in self.nodes.items():
            thumb = node.image
            if thumb.shape[1] > 240:
                scale = 240.0 / thumb.shape[1]
                thumb = cv2.resize(thumb, (240, max(1, int(thumb.shape[0] * scale))))
            cv2.imwrite(os.path.join(node_dir, f"node_{node_id:04d}.jpg"), thumb)

        data = self.to_debug_dict()
        json_path = os.path.join(output_dir, "topo_map.json")
        html_path = os.path.join(output_dir, "index.html")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._build_debug_html(data))

        return {"json": json_path, "html": html_path, "nodes_dir": node_dir}

    def status_str(self) -> str:
        loops = self._loop_closure_count()
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
