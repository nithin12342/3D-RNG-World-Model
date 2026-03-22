"""
Agentic Sandbox for Tool Execution
Chief AGI Architect & Neurosymbolic Integrator Implementation

This module implements:
- AgentEnvironment: Tied to the LLM Tool Executor
- Executable tool functions: Return unstructured data to be parsed into structured KG triples

References:
- [cite: 8] KGoT - dual-executor for graph management and tool execution
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json
import tempfile
import os


class ToolType(Enum):
    """Types of tools available in the sandbox."""
    PYTHON = "python"
    SEARCH = "search"
    FILE = "file"
    COMPUTATION = "computation"
    DATA = "data"


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    tool_name: str
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentEnvironment:
    """
    Agent environment tied to the LLM Tool Executor.
    Manages tool execution and returns unstructured data.
    """
    
    def __init__(self, working_dir: Optional[str] = None):
        """
        Initialize the Agent Environment.
        
        Args:
            working_dir: Working directory for file operations
        """
        self.working_dir = working_dir or tempfile.mkdtemp()
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self.execution_history: List[ToolResult] = []
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tool functions."""
        self.register_tool("execute_python", self.execute_python, "Execute Python code")
        self.register_tool("browse_web", self.browse_web, "Search the web")
        self.register_tool("read_file", self.read_file, "Read a file")
        self.register_tool("write_file", self.write_file, "Write to a file")
        self.register_tool("list_directory", self.list_directory, "List directory contents")
        self.register_tool("compute", self.compute, "Perform computation")
    
    def register_tool(self, name: str, function: Callable, description: str = ""):
        """
        Register a new tool.
        
        Args:
            name: Tool name
            function: Tool function
            description: Tool description
        """
        self.tools[name] = function
        self.tool_descriptions[name] = description
    
    def get_tool_description(self, name: str) -> str:
        """Get the description for a registered tool."""
        return self.tool_descriptions.get(name, "")
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        import time
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                output=None,
                error=f"Tool '{tool_name}' not found",
                execution_time=time.time() - start_time
            )
        
        try:
            result = self.tools[tool_name](**kwargs)
            execution_time = time.time() - start_time
            
            tool_result = ToolResult(
                success=True,
                tool_name=tool_name,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tool_result = ToolResult(
                success=False,
                tool_name=tool_name,
                output=None,
                error=str(e),
                execution_time=execution_time
            )
        
        self.execution_history.append(tool_result)
        return tool_result
    
    # === Tool Implementations ===
    
    def execute_python(self, code: str) -> str:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_dir
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Execution timeout"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def browse_web(self, query: str, num_results: int = 5) -> str:
        """
        Search the web (placeholder - requires API).
        
        Args:
            query: Search query
            num_results: Number of results
            
        Returns:
            Search results
        """
        # This is a placeholder - in production would use web search API
        return f"Web search results for '{query}': [Placeholders for {num_results} results]"
    
    def read_file(self, path: str) -> str:
        """
        Read a file.
        
        Args:
            path: File path
            
        Returns:
            File contents
        """
        try:
            full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
            with open(full_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, path: str, content: str) -> str:
        """
        Write to a file.
        
        Args:
            path: File path
            content: Content to write
            
        Returns:
            Success message
        """
        try:
            full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def list_directory(self, path: str = ".") -> str:
        """
        List directory contents.
        
        Args:
            path: Directory path
            
        Returns:
            Directory listing
        """
        try:
            full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
            if not os.path.exists(full_path):
                return f"Directory does not exist: {path}"
            
            entries = os.listdir(full_path)
            result = f"Contents of {path}:\n"
            for entry in entries:
                full_entry = os.path.join(full_path, entry)
                if os.path.isdir(full_entry):
                    result += f"  [DIR] {entry}\n"
                else:
                    result += f"  [FILE] {entry}\n"
            return result
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def compute(self, expression: str) -> str:
        """
        Perform computation.
        
        Args:
            expression: Math expression
            
        Returns:
            Computation result
        """
        try:
            # Safe evaluation of mathematical expressions
            allowed_ops = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'sqrt': np.sqrt, 'abs': np.abs, 'log': np.log,
                'exp': np.exp, 'pi': np.pi, 'e': np.e,
                'sum': sum, 'min': min, 'max': max, 'pow': pow
            }
            
            # Replace common math functions
            expr = expression
            for op_name, op_func in allowed_ops.items():
                expr = expr.replace(op_name, f"allowed_ops['{op_name}']")
            
            result = eval(expr, {"__builtins__": {}}, allowed_ops)
            return str(result)
        except Exception as e:
            return f"Error computing: {str(e)}"
    
    def get_history(self) -> List[ToolResult]:
        """Get execution history."""
        return self.execution_history
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []


class ToolOutputParser:
    """
    Parses unstructured tool outputs into structured KG triples.
    """
    
    def __init__(self):
        """Initialize the output parser."""
        self.parse_patterns = [
            (r"(\w+)\s+is\s+(\w+)", "is_a"),
            (r"(\w+)\s+relates\s+to\s+(\w+)", "relates_to"),
            (r"(\w+)\s+causes\s+(\w+)", "causes"),
            (r"(\w+)\s+belongs\s+to\s+(\w+)", "belongs_to"),
        ]
    
    def parse(self, tool_output: str) -> List[Tuple[str, str, str]]:
        """
        Parse tool output into KG triples.
        
        Args:
            tool_output: Unstructured tool output
            
        Returns:
            List of (subject, relation, object) triples
        """
        triples = []
        
        for pattern, relation in self.parse_patterns:
            import re
            matches = re.finditer(pattern, tool_output.lower())
            for match in matches:
                subject, obj = match.groups()
                triples.append((subject, relation, obj))
        
        return triples
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction - capital words
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))


class AgenticSandbox:
    """
    Main class integrating agent environment and tool execution.
    
    All methods are wrapped in strict try/except blocks to prevent physics loop crashes.
    Returns structured dictionaries instead of raising exceptions.
    """
    
    def __init__(self, working_dir: Optional[str] = None):
        """Initialize the Agentic Sandbox."""
        self.environment = AgentEnvironment(working_dir)
        self.parser = ToolOutputParser()
        
        # KG triple storage
        self.kg_triples: List[Tuple[str, str, str]] = []
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool and parse results.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments
            
        Returns:
            Dictionary with 'tool', 'success', 'output' or 'error' keys.
            This ensures NO exceptions propagate to the physics engine!
        """
        try:
            result = self.environment.execute(tool_name, **kwargs)
            
            # Parse output into KG triples
            if result.success and result.output:
                try:
                    triples = self.parser.parse(str(result.output))
                    self.kg_triples.extend(triples)
                except Exception as parse_error:
                    # Don't let parsing errors crash the physics loop
                    return {
                        "tool": tool_name,
                        "success": True,  # Tool executed OK
                        "output": result.output,
                        "warning": f"KG parsing failed: {str(parse_error)}"
                    }
            
            # Return structured dictionary format
            return {
                "tool": tool_name,
                "success": result.success,
                "output": result.output if result.success else None,
                "error": result.error if not result.success else None
            }
            
        except Exception as e:
            # CRITICAL: Never let exceptions escape to the physics engine!
            # Return a structured error dictionary instead
            return {
                "tool": tool_name,
                "success": False,
                "output": None,
                "error": str(e)
            }
    
    def execute_tool_safe(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Alias for execute_tool with explicit safe naming.
        
        Returns:
            Structured dictionary: {"tool": name, "success": bool, "output": result_or_None, "error": error_msg_or_None}
        """
        return self.execute_tool(tool_name, **kwargs)
    
    def get_kg_triples(self) -> Dict[str, Any]:
        """
        Get all accumulated KG triples.
        
        Returns:
            Dictionary with 'triples' and 'success' keys.
        """
        try:
            return {
                "success": True,
                "triples": self.kg_triples,
                "count": len(self.kg_triples)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "triples": [],
                "count": 0
            }
    
    def clear_triples(self) -> Dict[str, Any]:
        """
        Clear accumulated KG triples.
        
        Returns:
            Dictionary with 'success' key.
        """
        try:
            self.kg_triples = []
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tool executions.
        
        Returns:
            Dictionary with execution statistics.
        """
        try:
            history = self.environment.get_history()
            
            return {
                "success": True,
                "summary": {
                    "total_executions": len(history),
                    "successful": sum(1 for h in history if h.success),
                    "failed": sum(1 for h in history if not h.success),
                    "kg_triples_extracted": len(self.kg_triples)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": {
                    "total_executions": 0,
                    "successful": 0,
                    "failed": 0,
                    "kg_triples_extracted": 0
                }
            }
