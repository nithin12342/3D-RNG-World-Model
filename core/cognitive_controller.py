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
            grs: Graph Reasoning State to manage\n            goo: Graph of Operations for execution tracking\n        \"\"\"\n        self.grs = grs\n        self.goo = goo\n        \n        # Knowledge graph for storage (can be NetworkX or Cypher-based)\n        self.kg = nx.MultiDiGraph()\n        \n        # Query patterns\n        self.query_buffer: List[str] = []\n    \n    def analyze_state(self) -> Dict[str, Any]:\n        \"\"\"\n        Analyze current GRS state and determine next actions.\n        \n        Returns:\n            Analysis results with recommended actions\n        \"\"\"\n        active_thoughts = self.grs.get_active_thoughts()\n        completed_thoughts = self.grs.get_completed_thoughts()\n        \n        analysis = {\n            'num_active': len(active_thoughts),\n            'num_completed': len(completed_thoughts),\n            'needs_retrieval': False,\n            'needs_tool_call': False,\n            'needs_synthesis': False,\n            'missing_info': []\n        }\n        \n        # Check if we need more information\n        for thought in active_thoughts:\n            if thought.thought_type == ThoughtType.QUESTION:\n                # Check if answer exists\n                has_answer = any(t.thought_type == ThoughtType.ANSWER and \n                               t.parent_ids == [thought.id] \n                               for t in completed_thoughts)\n                if not has_answer:\n                    analysis['needs_retrieval'] = True\n                    analysis['missing_info'].append(thought.id)\n            \n            elif thought.thought_type == ThoughtType.REASONING:\n                # Check if we need tool execution\n                if 'requires_tool' in thought.metadata:\n                    analysis['needs_tool_call'] = True\n        \n        # Check if final synthesis is needed\n        if len(completed_thoughts) >= 3:\n            analysis['needs_synthesis'] = True\n        \n        return analysis\n    \n    def formulate_query(self, question: str) -> str:\n        \"\"\"\n        Formulate a knowledge graph query from a question.\n        \n        Args:\n            question: The question to query\n            \n        Returns:\n            Formulated query string\n        \"\"\"\n        # Simple keyword-based query formulation\n        query = f\"MATCH (a) WHERE a.content CONTAINS '{question}' RETURN a\"\n        self.query_buffer.append(query)\n        return query\n    \n    def execute_graph_operation(self, operation: str, params: Dict[str, Any]) -> str:\n        \"\"\"\n        Execute a graph management operation.\n        \n        Args:\n            operation: The operation to execute\n            params: Operation parameters\n            \n        Returns:\n            Operation result\n        \"\"\"\n        if operation == "create_thought":\n            node_id = self.grs.create_node(\n                content=params['content'],\n                thought_type=ThoughtType(params['thought_type']),\n                parent_ids=params.get('parent_ids')\n            )\n            return node_id\n        \n        elif operation == "update_status":\n            self.grs.update_node_status(\n                node_id=params['node_id'],\n                status=NodeStatus(params['status']),\n                metadata=params.get('metadata')\n            )\n            return \"updated\"\n        \n        elif operation == "add_to_kg":\n            # Add node to knowledge graph\n            self.kg.add_node(params['node_id'], **params.get('attributes', {}))\n            if 'relations' in params:\n                for rel in params['relations']:\n                    self.kg.add_edge(rel['from'], rel['to'], **rel.get('attributes', {}))\n            return \"added_to_kg\"\n        \n        return \"unknown_operation\"\n    \n    def plan_next_steps(self) -> List[Dict[str, Any]]:\n        \"\"\"\n        Plan the next steps based on current state.\n        \n        Returns:\n            List of planned operations\n        \"\"\"\n        analysis = self.analyze_state()\n        plans = []\n        \n        if analysis['needs_retrieval']:\n            for missing_id in analysis['missing_info'][:2]:  # Limit to 2\n                plans.append({\n                    'operation': 'create_thought',\n                    'params': {\n                        'content': f'Retrieve information for thought {missing_id}',\n                        'thought_type': 'retrieval',\n                        'parent_ids': [missing_id]\n                    },\n                    'priority': 10\n                })\n        \n        if analysis['needs_tool_call']:\n            plans.append({\n                'operation': 'create_thought',\n                'params': {\n                    'content': 'Execute tool for reasoning',\n                    'thought_type': 'tool_call',\n                },\n                'priority': 8\n            })\n        \n        if analysis['needs_synthesis']:\n            plans.append({\n                'operation': 'create_thought',\n                'params': {\n                    'content': 'Synthesize all reasoning into final answer',\n                    'thought_type': 'synthesis',\n                },\n                'priority': 5\n            })\n        \n        return plans\n\n\nclass LLMToolExecutor:\n    \"\"\"\n    Invokes external tools based on the Graph Executor's plan.\n    Part of the KGoT dual-executor architecture.\n    \"\"\"\n    \n    def __init__(self, tool_registry: Optional[Dict[str, callable]] = None):\n        \"\"\"\n        Initialize LLM Tool Executor.\n        \n        Args:\n            tool_registry: Dictionary of available tools {name: function}\n        \"\"\"\n        self.tool_registry = tool_registry or {}\n        self.execution_history: List[Dict[str, Any]] = []\n    \n    def register_tool(self, name: str, function: callable, description: str = \"\"):\n        \"\"\"\n        Register a new tool.\n        \n        Args:\n            name: Tool name\n            function: Tool function\n            description: Tool description\n        \"\"\"\n        self.tool_registry[name] = function\n    \n    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:\n        \"\"\"\n        Execute a tool.\n        \n        Args:\n            tool_name: Name of the tool to execute\n            **kwargs: Tool arguments\n            \n        Returns:\n            Tool execution result\n        \"\"\"\n        if tool_name not in self.tool_registry:\n            return {\n                'success': False,\n                'error': f'Tool {tool_name} not found',\n                'result': None\n            }\n        \n        try:\n            result = self.tool_registry[tool_name](**kwargs)\n            self.execution_history.append({\n                'tool': tool_name,\n                'args': kwargs,\n                'result': result,\n                'success': True\n            })\n            return {\n                'success': True,\n                'result': result\n            }\n        except Exception as e:\n            self.execution_history.append({\n                'tool': tool_name,\n                'args': kwargs,\n                'error': str(e),\n                'success': False\n            })\n            return {\n                'success': False,\n                'error': str(e),\n                'result': None\n            }\n    \n    def parse_tool_result(self, result: Dict[str, Any]) -> str:\n        \"\"\"\n        Parse tool result into structured format for KG.\n        \n        Args:\n            result: Raw tool result\n            \n        Returns:\n            Parsed result string\n        \"\"\"\n        if not result.get('success'):\n            return f\"Tool execution failed: {result.get('error')}\"\n        \n        return str(result.get('result', ''))\n\n\nclass CognitiveController:\n    \"\"\"\n    Main cognitive controller combining GoT and KGoT.\n    Orchestrates the dual-executor architecture.\n    \"\"\"\n    \n    def __init__(self):\n        \"\"\"Initialize the Cognitive Controller.\"\"\"\n        # Initialize components\n        self.grs = GraphReasoningState()\n        self.goo = GraphOfOperations(self.grs)\n        self.graph_executor = LLMGraphExecutor(self.grs, self.goo)\n        self.tool_executor = LLMToolExecutor()\n        \n        # State\n        self.current_task: Optional[str] = None\n        self.is_running = False\n    \n    def start_task(self, task: str):\n        \"\"\"\n        Start a new reasoning task.\n        \n        Args:\n            task: The task/question to reason about\n        \"\"\"\n        self.current_task = task\n        self.is_running = True\n        \n        # Create initial thought\n        root_id = self.grs.create_node(\n            content=task,\n            thought_type=ThoughtType.QUESTION\n        )\n        \n        # Plan next steps\n        plans = self.graph_executor.plan_next_steps()\n        for plan in plans:\n            self.goo.add_operation(plan['node_id'], plan['operation'], plan['priority'])\n    \n    def step(self) -> bool:\n        \"\"\"\n        Execute one step of reasoning.\n        \n        Returns:\n            True if more steps are needed, False if complete\n        \"\"\"\n        if not self.is_running:\n            return False\n        \n        # Get next operation\n        op = self.goo.get_next_operation()\n        if not op:\n            # Plan new operations\n            plans = self.graph_executor.plan_next_steps()\n            if not plans:\n                self.is_running = False\n                return False\n            \n            for plan in plans:\n                node_id = self.graph_executor.execute_graph_operation(\n                    plan['operation'], plan['params']\n                )\n                self.goo.add_operation(node_id, plan['operation'], plan['priority'])\n            return True\n        \n        # Execute the operation\n        self.graph_executor.execute_graph_operation(\n            op['operation'], \n            {'node_id': op['node_id'], 'status': 'processing'}\n        )\n        \n        return True\n    \n    def run(self, max_steps: int = 100) -> str:\n        \"\"\"\n        Run the reasoning process.\n        \n        Args:\n            max_steps: Maximum number of reasoning steps\n            \n        Returns:\n            Final aggregated result\n        \"\"\"\n        self.start_task(self.current_task or \"\")\n        \n        for _ in range(max_steps):\n            if not self.step():\n                break\n        \n        return self.grs.aggregate_thoughts()\n    \n    def get_state(self) -> Dict[str, Any]:\n        \"\"\"\n        Get current controller state.\n        \n        Returns:\n            Current state dictionary\n        \"\"\"\n        return {\n            'task': self.current_task,\n            'is_running': self.is_running,\n            'num_thoughts': len(self.grs.nodes),\n            'active_thoughts': len(self.grs.get_active_thoughts()),\n            'completed_thoughts': len(self.grs.get_completed_thoughts())\n        }
