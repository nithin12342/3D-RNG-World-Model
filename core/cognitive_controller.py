"""
Cognitive Controller with Graph of Thoughts (GoT) and KGoT Dual-Executors
Chief AGI Architect & Neurosymbolic Integrator Implementation

This module implements:
- GraphReasoningState (GRS): Maintains history and states of all LLM thoughts
- GraphOfOperations (GoO): Tracks and aggregates non-linear thoughts
- LLM Graph Executor: Parses GRS, determines missing info, formulates graph queries
- LLM Tool Executor: Invokes external tools based on Graph Executor's plan

References:
- [cite: 9] Graph of Thoughts (GoT) - arbitrary graph reasoning
- [cite: 8] KGoT - dual-executor for graph management and tool execution
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx


class ThoughtType(Enum):
    """Types of thoughts in the graph."""
    QUESTION = "question"
    ANSWER = "answer"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYNTHESIS = "synthesis"
    FINAL = "final"


class NodeStatus(Enum):
    """Status of thought nodes."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ThoughtNode:
    """Represents a single thought in the reasoning graph."""
    id: str
    content: str
    thought_type: ThoughtType
    status: NodeStatus = NodeStatus.PENDING
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    
    def __hash__(self):
        return hash(self.id)


class GraphReasoningState:
    """
    Maintains the history and states of all LLM thoughts.
    Implements the Graph of Thoughts (GoT) paradigm.
    """
    
    def __init__(self, max_nodes: int = 1000):
        """
        Initialize Graph Reasoning State.
        
        Args:
            max_nodes: Maximum number of thought nodes to maintain
        """
        self.max_nodes = max_nodes
        self.nodes: Dict[str, ThoughtNode] = {}
        self.graph = nx.DiGraph()  # Directed graph for thought dependencies
        self.root_ids: List[str] = []  # Root thought nodes
        self.leaf_ids: List[str] = []  # Leaf thought nodes (pending completion)
        
        # Node ID counter
        self._node_counter = 0
        
        # Reasoning path stack (for RTSOG integration)
        self.reasoning_stack: deque = deque(maxlen=100)
    
    def create_node(self, content: str, thought_type: ThoughtType, 
                   parent_ids: Optional[List[str]] = None) -> str:
        """
        Create a new thought node.
        
        Args:
            content: The thought content
            thought_type: Type of the thought
            parent_ids: IDs of parent thoughts
            
        Returns:
            The new node ID
        """
        node_id = f"thought_{self._node_counter}"
        self._node_counter += 1
        
        node = ThoughtNode(
            id=node_id,
            content=content,
            thought_type=thought_type,
            parent_ids=parent_ids or [],
            status=NodeStatus.PENDING
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        
        # Link to parents
        if parent_ids:
            for parent_id in parent_ids:
                if parent_id in self.nodes:
                    self.graph.add_edge(parent_id, node_id)
                    self.nodes[parent_id].children_ids.append(node_id)
        else:
            self.root_ids.append(node_id)
        
        # Update leaf nodes
        self._update_leaf_nodes()
        
        # Enforce max nodes limit
        if len(self.nodes) > self.max_nodes:
            self._prune_oldest()
        
        return node_id
    
    def _update_leaf_nodes(self):
        """Update the list of leaf nodes."""
        self.leaf_ids = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
    
    def _prune_oldest(self):
        """Prune oldest nodes to maintain max_nodes limit."""
        # Remove oldest non-root nodes
        nodes_to_remove = []
        for node_id in self.root_ids:
            if len(self.nodes) <= self.max_nodes:
                break
            # Find oldest leaf nodes
            for leaf_id in self.leaf_ids:
                if leaf_id not in nodes_to_remove:
                    nodes_to_remove.append(leaf_id)
                    if len(self.nodes) - len(nodes_to_remove) <= self.max_nodes:
                        break
        
        for node_id in nodes_to_remove[:len(nodes_to_remove) - self.max_nodes + len(self.nodes)]:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.graph.remove_node(node_id)
        
        self._update_leaf_nodes()
        self.root_ids = [n for n in self.root_ids if n in self.nodes]
    
    def update_node_status(self, node_id: str, status: NodeStatus, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Update node status and metadata."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            if metadata:
                self.nodes[node_id].metadata.update(metadata)
            
            if status == NodeStatus.COMPLETED:
                self._update_leaf_nodes()
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get a thought node by ID."""
        return self.nodes.get(node_id)
    
    def get_active_thoughts(self) -> List[ThoughtNode]:
        """Get all pending/processing thoughts."""
        return [n for n in self.nodes.values() 
                if n.status in [NodeStatus.PENDING, NodeStatus.PROCESSING]]
    
    def get_completed_thoughts(self) -> List[ThoughtNode]:
        """Get all completed thoughts."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.COMPLETED]
    
    def get_reasoning_path(self, node_id: str) -> List[str]:
        """Get the reasoning path from root to given node."""
        if node_id not in self.nodes:
            return []
        
        try:
            path = nx.shortest_path(self.graph, self.root_ids[0], node_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def aggregate_thoughts(self) -> str:
        """Aggregate all completed thoughts into a final response."""
        completed = self.get_completed_thoughts()
        if not completed:
            return ""
        
        # Sort by topological order
        try:
            topo_order = list(nx.topological_sort(self.graph))
            completed.sort(key=lambda x: topo_order.index(x.id) if x.id in topo_order else -1)
        except nx.NetworkXError:
            pass
        
        # Concatenate thoughts
        aggregated = "\n\n".join([f"[{t.thought_type.value}] {t.content}" 
                                   for t in completed])
        
        return aggregated


class GraphOfOperations:
    """
    Tracks and aggregates non-linear thoughts.
    Manages the execution flow of the reasoning graph.
    """
    
    def __init__(self, grs: GraphReasoningState):
        """
        Initialize Graph of Operations.
        
        Args:
            grs: The Graph Reasoning State to manage
        """
        self.grs = grs
        self.execution_queue: deque = deque()
        self.completed_operations: List[str] = []
        
    def add_operation(self, node_id: str, operation: str, priority: int = 0):
        """Add an operation to the execution queue."""
        self.execution_queue.append({
            'node_id': node_id,
            'operation': operation,
            'priority': priority
        })
        
        # Sort by priority (higher first)
        self.execution_queue = deque(
            sorted(self.execution_queue, key=lambda x: -x['priority'])
        )
    
    def get_next_operation(self) -> Optional[Dict[str, Any]]:
        """Get the next operation to execute."""
        if self.execution_queue:
            return self.execution_queue.popleft()
        return None
    
    def is_complete(self) -> bool:
        """Check if all operations are complete."""
        return len(self.execution_queue) == 0


class LLMGraphExecutor:
    """
    Parses the GRS, determines missing information, and formulates graph queries.
    Part of the KGoT dual-executor architecture.
    """
    
    def __init__(self, grs: GraphReasoningState, goo: GraphOfOperations):
        """
        Initialize LLM Graph Executor.
        
        Args:
            grs: Graph Reasoning State to manage
            goo: Graph of Operations for execution tracking
        """
        self.grs = grs
        self.goo = goo
        
        # Knowledge graph for storage (can be NetworkX or Cypher-based)
        self.kg = nx.MultiDiGraph()
        
        # Query patterns
        self.query_buffer: List[str] = []
    
    def analyze_state(self) -> Dict[str, Any]:
        """
        Analyze current GRS state and determine next actions.
        
        Returns:
            Analysis results with recommended actions
        """
        active_thoughts = self.grs.get_active_thoughts()
        completed_thoughts = self.grs.get_completed_thoughts()
        
        analysis = {
            'num_active': len(active_thoughts),
            'num_completed': len(completed_thoughts),
            'needs_retrieval': False,
            'needs_tool_call': False,
            'needs_synthesis': False,
            'missing_info': []
        }
        
        # Check if we need more information
        for thought in active_thoughts:
            if thought.thought_type == ThoughtType.QUESTION:
                # Check if answer exists
                has_answer = any(t.thought_type == ThoughtType.ANSWER and 
                               t.parent_ids == [thought.id] 
                               for t in completed_thoughts)
                if not has_answer:
                    analysis['needs_retrieval'] = True
                    analysis['missing_info'].append(thought.id)
            
            elif thought.thought_type == ThoughtType.REASONING:
                # Check if we need tool execution
                if 'requires_tool' in thought.metadata:
                    analysis['needs_tool_call'] = True
        
        # Check if final synthesis is needed
        if len(completed_thoughts) >= 3:
            analysis['needs_synthesis'] = True
        
        return analysis
    
    def formulate_query(self, question: str) -> str:
        """
        Formulate a knowledge graph query from a question.
        
        Args:
            question: The question to query
            
        Returns:
            Formulated query string
        """
        # Simple keyword-based query formulation
        query = f"MATCH (a) WHERE a.content CONTAINS '{question}' RETURN a"
        self.query_buffer.append(query)
        return query
    
    def execute_graph_operation(self, operation: str, params: Dict[str, Any]) -> str:
        """
        Execute a graph management operation.
        
        Args:
            operation: The operation to execute
            params: Operation parameters
            
        Returns:
            Operation result
        """
        if operation == "create_thought":
            node_id = self.grs.create_node(
                content=params['content'],
                thought_type=ThoughtType(params['thought_type']),
                parent_ids=params.get('parent_ids')
            )
            return node_id
        
        elif operation == "update_status":
            self.grs.update_node_status(
                node_id=params['node_id'],
                status=NodeStatus(params['status']),
                metadata=params.get('metadata')
            )
            return "updated"
        
        elif operation == "add_to_kg":
            # Add node to knowledge graph
            self.kg.add_node(params['node_id'], **params.get('attributes', {}))
            if 'relations' in params:
                for rel in params['relations']:
                    self.kg.add_edge(rel['from'], rel['to'], **rel.get('attributes', {}))
            return "added_to_kg"
        
        return "unknown_operation"
    
    def plan_next_steps(self) -> List[Dict[str, Any]]:
        """
        Plan the next steps based on current state.
        
        Returns:
            List of planned operations
        """
        analysis = self.analyze_state()
        plans = []
        
        if analysis['needs_retrieval']:
            for missing_id in analysis['missing_info'][:2]:  # Limit to 2
                plans.append({
                    'operation': 'create_thought',
                    'params': {
                        'content': f'Retrieve information for thought {missing_id}',
                        'thought_type': 'retrieval',
                        'parent_ids': [missing_id]
                    },
                    'priority': 10
                })
        
        if analysis['needs_tool_call']:
            plans.append({
                'operation': 'create_thought',
                'params': {
                    'content': 'Execute tool for reasoning',
                    'thought_type': 'tool_call',
                },
                'priority': 8
            })
        
        if analysis['needs_synthesis']:
            plans.append({
                'operation': 'create_thought',
                'params': {
                    'content': 'Synthesize all reasoning into final answer',
                    'thought_type': 'synthesis',
                },
                'priority': 5
            })
        
        return plans


class LLMToolExecutor:
    """
    Invokes external tools based on the Graph Executor's plan.
    Part of the KGoT dual-executor architecture.
    """
    
    def __init__(self, tool_registry: Optional[Dict[str, callable]] = None):
        """
        Initialize LLM Tool Executor.
        
        Args:
            tool_registry: Dictionary of available tools {name: function}
        """
        self.tool_registry = tool_registry or {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, function: callable, description: str = ""):
        """
        Register a new tool.
        
        Args:
            name: Tool name
            function: Tool function
            description: Tool description
        """
        self.tool_registry[name] = function
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tool_registry:
            return {
                'success': False,
                'error': f'Tool {tool_name} not found',
                'result': None
            }
        
        try:
            result = self.tool_registry[tool_name](**kwargs)
            self.execution_history.append({
                'tool': tool_name,
                'args': kwargs,
                'result': result,
                'success': True
            })
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            self.execution_history.append({
                'tool': tool_name,
                'args': kwargs,
                'error': str(e),
                'success': False
            })
            return {
                'success': False,
                'error': str(e),
                'result': None
            }
    
    def parse_tool_result(self, result: Dict[str, Any]) -> str:
        """
        Parse tool result into structured format for KG.
        
        Args:
            result: Raw tool result
            
        Returns:
            Parsed result string
        """
        if not result.get('success'):
            return f"Tool execution failed: {result.get('error')}"
        
        return str(result.get('result', ''))


class CognitiveController:
    """
    Main cognitive controller combining GoT and KGoT.
    Orchestrates the dual-executor architecture.
    """
    
    def __init__(self):
        """Initialize the Cognitive Controller."""
        # Initialize components
        self.grs = GraphReasoningState()
        self.goo = GraphOfOperations(self.grs)
        self.graph_executor = LLMGraphExecutor(self.grs, self.goo)
        self.tool_executor = LLMToolExecutor()
        
        # State
        self.current_task: Optional[str] = None
        self.is_running = False
    
    def start_task(self, task: str):
        """
        Start a new reasoning task.
        
        Args:
            task: The task/question to reason about
        """
        self.current_task = task
        self.is_running = True
        
        # Create initial thought
        root_id = self.grs.create_node(
            content=task,
            thought_type=ThoughtType.QUESTION
        )
        
        # Plan next steps
        plans = self.graph_executor.plan_next_steps()
        for plan in plans:
            self.goo.add_operation(plan['node_id'], plan['operation'], plan['priority'])
    
    def step(self) -> bool:
        """
        Execute one step of reasoning.
        
        Returns:
            True if more steps are needed, False if complete
        """
        if not self.is_running:
            return False
        
        # Get next operation
        op = self.goo.get_next_operation()
        if not op:
            # Plan new operations
            plans = self.graph_executor.plan_next_steps()
            if not plans:
                self.is_running = False
                return False
            
            for plan in plans:
                node_id = self.graph_executor.execute_graph_operation(
                    plan['operation'], plan['params']
                )
                self.goo.add_operation(node_id, plan['operation'], plan['priority'])
            return True
        
        # Execute the operation
        self.graph_executor.execute_graph_operation(
            op['operation'], 
            {'node_id': op['node_id'], 'status': 'processing'}
        )
        
        return True
    
    def run(self, max_steps: int = 100) -> str:
        """
        Run the reasoning process.
        
        Args:
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Final aggregated result
        """
        self.start_task(self.current_task or "")
        
        for _ in range(max_steps):
            if not self.step():
                break
        
        return self.grs.aggregate_thoughts()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current controller state.
        
        Returns:
            Current state dictionary
        """
        return {
            'task': self.current_task,
            'is_running': self.is_running,
            'num_thoughts': len(self.grs.nodes),
            'active_thoughts': len(self.grs.get_active_thoughts()),
            'completed_thoughts': len(self.grs.get_completed_thoughts())
        }
