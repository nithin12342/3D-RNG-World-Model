"""
Neurosymbolic Knowledge Graph with RTSOG SC-MCTS for X=4 Plane
Chief AGI Architect & Neurosymbolic Integrator Implementation

This module implements:
- Knowledge Graph for X=4 plane reasoning
- SC-MCTS (Self-Critic Monte Carlo Tree Search) for reasoning path evaluation
- Question decomposition for complex queries
- Reasoning Path Stack for guiding final generation

References:
- [cite: 7] RTSOG - Self-Critic Monte Carlo Tree Search
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    ENTITY = "entity"
    RELATION = "relation"
    CONCEPT = "concept"
    QUERY = "query"
    ANSWER = "answer"


@dataclass
class KGNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: NodeType
    content: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    visit_count: int = 0
    total_reward: float = 0.0


@dataclass
class KGEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


class ReasoningPath:
    """Represents a reasoning path through the knowledge graph."""
    
    def __init__(self, nodes: List[KGNode], edges: List[KGEdge], score: float = 0.0):
        self.nodes = nodes
        self.edges = edges
        self.score = score
        self.length = len(nodes)
    
    def __repr__(self):
        return f"ReasoningPath(length={self.length}, score={self.score:.3f})"
    
    def __lt__(self, other):
        return self.score < other.score


class KnowledgeGraph:
    """
    Knowledge Graph for the X=4 plane.
    Stores entities, relations, and concepts for reasoning.
    """
    
    def __init__(self):
        """Initialize the Knowledge Graph."""
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self.adjacency: Dict[str, List[str]] = {}
        
        # Node ID counter
        self._node_counter = 0
    
    def add_node(self, node_type: NodeType, content: str, 
                attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the knowledge graph.
        
        Args:
            type: Type of the node
            content: Node content
            attributes: Optional node attributes
            
        Returns:
            The new node ID
        """
        node_id = f"kg_node_{self._node_counter}"
        self._node_counter += 1
        
        node = KGNode(
            id=node_id,
            type=node_type,
            content=content,
            attributes=attributes or {}
        )
        
        self.nodes[node_id] = node
        self.adjacency[node_id] = []
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, 
                relation: str, weight: float = 1.0):
        """Add an edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        
        edge = KGEdge(source_id, target_id, relation, weight)
        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        return self.adjacency.get(node_id, [])
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def traverse(self, start_id: str, max_depth: int = 3) -> List[ReasoningPath]:
        """
        Traverse the graph from a starting node.
        
        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth
            
        Returns:
            List of reasoning paths
        """
        paths = []
        
        def dfs(node_id: str, current_path: List[KGNode], current_edges: List[KGEdge], depth: int):
            if depth >= max_depth:
                if current_path:
                    paths.append(ReasoningPath(current_path[:], current_edges[:]))
                return
            
            node = self.get_node(node_id)
            if not node:
                return
            
            current_path.append(node)
            
            for neighbor_id in self.get_neighbors(node_id):
                neighbor = self.get_node(neighbor_id)
                if neighbor:
                    # Find the edge
                    for edge in self.edges:
                        if edge.source_id == node_id and edge.target_id == neighbor_id:
                            current_edges.append(edge)
                            dfs(neighbor_id, current_path, current_edges, depth + 1)
                            current_edges.pop()
            
            current_path.pop()
        
        dfs(start_id, [], [], 0)
        return paths


class RewardModel:
    """
    Self-Critic reward model for evaluating reasoning paths.
    Uses heuristics to score path quality.
    """
    
    def __init__(self):
        """Initialize the reward model."""
        self.reward_history: List[float] = []
    
    def evaluate(self, path: ReasoningPath, query: str, context: Dict[str, Any]) -> float:
        """
        Evaluate a reasoning path.
        
        Args:
            path: The reasoning path to evaluate
            query: The original query
            context: Additional context
            
        Returns:
            Reward score between 0 and 1
        """
        # Base reward from path score
        reward = path.score
        
        # Length bonus (prefer longer reasoning chains)
        length_bonus = min(path.length * 0.1, 0.3)
        reward += length_bonus
        
        # Relevance to query
        query_terms = set(query.lower().split())
        path_content = " ".join([n.content.lower() for n in path.nodes])
        path_terms = set(path_content.split())
        relevance = len(query_terms & path_terms) / max(len(query_terms), 1)
        reward += relevance * 0.3
        
        # Diversity bonus (encourage different reasoning approaches)
        diversity_bonus = 0.0
        if self.reward_history:
            # Penalize similar scores
            avg_past = np.mean(self.reward_history[-5:])
            score_diff = abs(reward - avg_past)
            diversity_bonus = min(score_diff * 0.2, 0.1)
        reward += diversity_bonus
        
        # Normalize to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        self.reward_history.append(reward)
        return reward


class SC_MCTS:
    """
    Self-Critic Monte Carlo Tree Search.
    Explores graph relations, evaluates candidate nodes using reward model,
    and emits EoS (End-of-Search) signal to prevent infinite loops.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, reward_model: RewardModel,
                 max_iterations: int = 100, exploration_constant: float = 1.414,
                 eos_threshold: float = 0.95):
        """
        Initialize SC-MCTS.
        
        Args:
            knowledge_graph: The knowledge graph to search
            reward_model: The reward model for evaluation
            max_iterations: Maximum number of search iterations
            exploration_constant: UCB exploration constant
            eos_threshold: Threshold for End-of-Search signal
        """
        self.kg = knowledge_graph
        self.reward_model = reward_model
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.eos_threshold = eos_threshold
        
        # Search state
        self.root_id: Optional[str] = None
        self.query: str = ""
        self.best_paths: List[ReasoningPath] = []
        self.iteration_count: int = 0
        
        # UCB values for nodes
        self.ucb_values: Dict[str, float] = {}
    
    def search(self, start_node_id: str, query: str, 
              context: Optional[Dict[str, Any]] = None) -> Tuple[List[ReasoningPath], bool]:
        """
        Perform MCTS search from a starting node.
        
        Args:
            start_node_id: Starting node ID
            query: The search query
            context: Additional context
            
        Returns:
            Tuple of (best reasoning paths, EoS signal)
        """
        self.root_id = start_node_id
        self.query = query
        self.best_paths = []
        self.iteration_count = 0
        self.ucb_values = {}
        
        context = context or {}
        
        for _ in range(self.max_iterations):
            self.iteration_count += 1
            
            # Selection
            path = self._select(self.root_id, [], [], 0)
            
            # Evaluation
            if path:
                reward = self.reward_model.evaluate(path, self.query, context)
                path.score = reward
                self.best_paths.append(path)
            
            # Check EoS condition
            if self.best_paths:
                best_score = max(p.score for p in self.best_paths)
                if best_score >= self.eos_threshold:
                    # EoS signal - stop searching
                    break
            
            # Limit iterations if no progress
            if len(self.best_paths) > 10 and self.iteration_count > 50:
                # Check if we're making progress
                recent_best = max(p.score for p in self.best_paths[-10:])
                if recent_best <= 0.01:
                    break
        
        # Sort by score and return top paths
        self.best_paths.sort(key=lambda x: -x.score)
        
        # Determine EoS
        eos = len(self.best_paths) > 0 and self.best_paths[0].score >= self.eos_threshold
        
        return self.best_paths[:10], eos
    
    def _select(self, node_id: str, current_path: List[KGNode],
               current_edges: List[KGEdge], depth: int) -> Optional[ReasoningPath]:
        """
        Select next node using UCB.
        """
        if depth >= 5:  # Max depth
            return ReasoningPath(current_path[:], current_edges[:]) if current_path else None
        
        node = self.kg.get_node(node_id)
        if not node:
            return None
        
        current_path.append(node)
        
        # Get neighbors
        neighbors = self.kg.get_neighbors(node_id)
        
        if not neighbors:
            # Leaf node - return current path
            result = ReasoningPath(current_path[:], current_edges[:]) if current_path else None
            current_path.pop()
            return result
        
        # Select best neighbor using UCB
        best_neighbor = None
        best_ucb = float('-inf')
        
        for neighbor_id in neighbors:
            ucb = self._ucb(neighbor_id)
            if ucb > best_ucb:
                best_ucb = ucb
                best_neighbor = neighbor_id
        
        if best_neighbor:
            # Find edge
            for edge in self.kg.edges:
                if edge.source_id == node_id and edge.target_id == best_neighbor:
                    current_edges.append(edge)
                    result = self._select(best_neighbor, current_path, current_edges, depth + 1)
                    current_edges.pop()
                    current_path.pop()
                    return result
        
        current_path.pop()
        return ReasoningPath(current_path[:], current_edges[:]) if current_path else None
    
    def _ucb(self, node_id: str) -> float:
        """Compute UCB value for a node."""
        node = self.kg.get_node(node_id)
        if not node:
            return 0.0
        
        # Exploitation
        if node.visit_count > 0:
            exploitation = node.total_reward / node.visit_count
        else:
            exploitation = 0.0
        
        # Exploration
        if node.visit_count > 0:
            exploration = self.exploration_constant * np.sqrt(
                np.log(self.iteration_count) / node.visit_count
            )
        else:
            exploration = 100.0  # High value for unvisited nodes
        
        return exploitation + exploration


class QuestionDecomposer:
    """
    Decomposes complex questions into simpler sub-questions.
    """
    
    def __init__(self):
        """Initialize the question decomposer."""
        self.decomposition_patterns = [
            ("What is X and Y?", ["What is X?", "What is Y?"]),
            ("How does X affect Y?", ["What is X?", "How does X work?", "What is Y?", "How does X relate to Y?"]),
            ("Why X because Y?", ["Why X?", "What causes Y?", "How does Y lead to X?"]),
        ]
    
    def decompose(self, question: str) -> List[str]:
        """
        Decompose a complex question into simpler sub-questions.
        
        Args:
            question: The complex question
            
        Returns:
            List of simpler sub-questions
        """
        question_lower = question.lower()
        
        for pattern, sub_questions in self.decomposition_patterns:
            if pattern.lower() in question_lower:
                return sub_questions
        
        # Default: return original question as single-element list
        # Could add more sophisticated decomposition here
        return [question]


class ReasoningPathStack:
    """
    Stack to store reasoning paths in descending order of weights.
    Guides final generation.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the reasoning path stack.
        
        Args:
            max_size: Maximum number of paths to store
        """
        self.max_size = max_size
        self.stack: List[ReasoningPath] = []
    
    def push(self, path: ReasoningPath):
        """Push a reasoning path onto the stack."""
        # Insert in descending order by score
        inserted = False
        for i, existing in enumerate(self.stack):
            if path.score > existing.score:
                self.stack.insert(i, path)
                inserted = True
                break
        
        if not inserted:
            self.stack.append(path)
        
        # Enforce max size
        if len(self.stack) > self.max_size:
            self.stack.pop()
    
    def push_multiple(self, paths: List[ReasoningPath]):
        """Push multiple reasoning paths."""
        for path in paths:
            self.push(path)
    
    def get_top_k(self, k: int) -> List[ReasoningPath]:
        """Get top K reasoning paths."""
        return self.stack[:k]
    
    def get_all(self) -> List[ReasoningPath]:
        """Get all reasoning paths."""
        return self.stack
    
    def clear(self):
        """Clear the stack."""
        self.stack = []


class NeurosymbolicKG:
    """
    Main class integrating all components for X=4 plane reasoning.
    Combines Knowledge Graph, SC-MCTS, and Reasoning Path Stack.
    """
    
    def __init__(self):
        """Initialize the Neurosymbolic Knowledge Graph."""
        # Components
        self.kg = KnowledgeGraph()
        self.reward_model = RewardModel()
        self.mcts = SC_MCTS(self.kg, self.reward_model)
        self.decomposer = QuestionDecomposer()
        self.reasoning_stack = ReasoningPathStack()
        
        # State
        self.current_query: Optional[str] = None
        self.search_results: List[ReasoningPath] = []
        self.eos_triggered: bool = False
    
    def decompose_question(self, question: str) -> List[str]:
        """
        Decompose a complex question into simpler sub-questions.
        
        Args:
            question: The question to decompose
            
        Returns:
            List of sub-questions
        """
        return self.decomposer.decompose(question)
    
    def search(self, query: str, start_node_id: Optional[str] = None,
              context: Optional[Dict[str, Any]] = None) -> Tuple[List[ReasoningPath], bool]:
        """
        Perform reasoning search using SC-MCTS.
        
        Args:
            query: The search query
            start_node_id: Optional starting node ID
            context: Additional context
            
        Returns:
            Tuple of (reasoning paths, EoS signal)
        """
        self.current_query = query
        
        # Use provided start node or find relevant node
        if not start_node_id:
            # Find most relevant node
            start_node_id = self._find_relevant_node(query)
        
        if not start_node_id:
            return [], False
        
        # Perform search
        paths, eos = self.mcts.search(start_node_id, query, context)
        
        # Store results
        self.search_results = paths
        self.eos_triggered = eos
        
        # Push to reasoning stack
        self.reasoning_stack.push_multiple(paths)
        
        return paths, eos
    
    def _find_relevant_node(self, query: str) -> Optional[str]:
        """Find the most relevant node for a query."""
        query_terms = set(query.lower().split())
        
        best_node_id = None
        best_score = 0.0
        
        for node_id, node in self.kg.nodes.items():
            node_terms = set(node.content.lower().split())
            score = len(query_terms & node_terms)
            if score > best_score:
                best_score = score
                best_node_id = node_id
        
        return best_node_id
    
    def get_best_paths(self, k: int = 5) -> List[ReasoningPath]:
        """
        Get the top K reasoning paths.
        
        Args:
            k: Number of paths to retrieve
            
        Returns:
            List of top K reasoning paths
        """
        return self.reasoning_stack.get_top_k(k)
    
    def add_knowledge(self, subject: str, relation: str, obj: str,
                     attributes: Optional[Dict[str, Any]] = None):
        """
        Add knowledge to the graph.
        
        Args:
            subject: Subject entity
            relation: Relation type
            obj: Object entity
            attributes: Optional attributes
        """
        # Add subject node
        subject_id = self.kg.add_node(NodeType.ENTITY, subject, attributes)
        
        # Add object node
        object_id = self.kg.add_node(NodeType.ENTITY, obj)
        
        # Add edge
        self.kg.add_edge(subject_id, object_id, relation)
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process."""
        if not self.search_results:
            return "No reasoning paths found."
        
        summary = f"Found {len(self.search_results)} reasoning paths.\n"
        summary += f"EoS triggered: {self.eos_triggered}\n\n"
        
        for i, path in enumerate(self.search_results[:5]):
            summary += f"Path {i+1} (score: {path.score:.3f}):\n"
            for node in path.nodes:
                summary += f"  - {node.content}\n"
            summary += "\n"
        
        return summary
