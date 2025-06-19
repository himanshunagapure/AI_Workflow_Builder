import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal, Callable, Union
from dataclasses import dataclass, asdict
import json
import re
from difflib import SequenceMatcher

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

from tool_registry import load_dynamic_tools

# Import all available tools dynamically
import importlib
import inspect
from pathlib import Path

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_tokens=500,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )
    print("✅ LLM configured successfully")
except Exception as e:
    print(f"❌ LLM configuration failed: {e}")
    raise

# ✅ Generic Context and Memory Management
@dataclass
class TaskContext:
    """Generic context for any task execution"""
    task_id: str
    user_query: str
    sub_tasks: List[str]
    completed_steps: List[Dict[str, Any]]
    intermediate_results: Dict[str, Any]
    created_at: datetime
    status: Literal["planning", "executing", "completed", "failed"] = "planning"

@dataclass
class ConfirmationRequest:
    """Generic confirmation request for any action"""
    id: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    context: Optional[TaskContext]
    created_at: datetime
    status: Literal["pending", "approved", "rejected", "executed"] = "pending"

@dataclass
class UserSession:
    """Generic user session for any type of interaction"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    current_context: Optional[TaskContext] = None
    pending_confirmations: List[ConfirmationRequest] = None
    execution_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.pending_confirmations is None:
            self.pending_confirmations = []
        if self.execution_history is None:
            self.execution_history = []

class SessionManager:
    """Generic session management for any domain"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self._sessions: Dict[str, UserSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Create a new session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now
        )
        
        self._sessions[session_id] = session
        self._cleanup_expired_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session and update last activity"""
        session = self._sessions.get(session_id)
        if not session:
            self.create_session(session_id=session_id)
            session = self._sessions[session_id]
        else:
            session.last_activity = datetime.now()
        return session
    
    def update_context(self, session_id: str, context: TaskContext):
        """Update task context for session"""
        session = self.get_session(session_id)
        if session:
            session.current_context = context
    
    def add_confirmation_request(self, session_id: str, action_type: str, 
                               description: str, parameters: Dict[str, Any]) -> str:
        """Add confirmation request to session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        confirmation = ConfirmationRequest(
            id=str(uuid.uuid4()),
            action_type=action_type,
            description=description,
            parameters=parameters,
            context=session.current_context,
            created_at=datetime.now()
        )
        
        session.pending_confirmations.append(confirmation)
        return confirmation.id
    
    def get_pending_confirmations(self, session_id: str) -> List[ConfirmationRequest]:
        """Get all pending confirmations for session"""
        session = self.get_session(session_id)
        if not session:
            return []
        return [req for req in session.pending_confirmations if req.status == "pending"]
    
    def update_confirmation_status(self, session_id: str, confirmation_id: str, status: str) -> bool:
        """Update confirmation status"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        for confirmation in session.pending_confirmations:
            if confirmation.id == confirmation_id:
                confirmation.status = status
                return True
        return False
    
    def add_execution_record(self, session_id: str, tool_name: str, 
                           inputs: Dict[str, Any], outputs: Any, success: bool):
        """Record tool execution"""
        session = self.get_session(session_id)
        if session:
            session.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "tool_name": tool_name,
                "inputs": inputs,
                "outputs": str(outputs),
                "success": success
            })
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        
        for sid in expired_sessions:
            del self._sessions[sid]
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        self._sessions.pop(session_id, None)

# ✅ Global Session Manager Instance
session_manager = SessionManager()

# ✅ LangGraph State Definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    user_query: str
    task_context: Optional[Dict[str, Any]]
    pending_confirmations: List[Dict[str, Any]]
    workflow_complete: bool
    intermediate_results: Dict[str, Any]
    execution_step: int  # Add execution step tracking

# ✅ Dynamic Tool Discovery and Loading
class ToolDiscovery:
    """Dynamic tool discovery and management"""
    
    def __init__(self):
        self.static_tools = []
        self.dynamic_tools = []
        self.all_tools = []
        self._load_all_tools()
    
    def _load_static_tools(self):
        """Load tools from tools/ directory"""
        tools_dir = Path("tools")
        if not tools_dir.exists():
            return []
        
        static_tools = []
        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            try:
                module_name = f"tools.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                # Get the main function from the module
                main_func = getattr(module, py_file.stem, None)
                
                if main_func and inspect.isfunction(main_func) and hasattr(main_func, '__doc__'):
                    # Create tool from the main function
                    tool = Tool(
                        name=py_file.stem,
                        func=main_func,
                        description=main_func.__doc__.strip() if main_func.__doc__ else f"Function {py_file.stem}"
                    )
                    static_tools.append(tool)
                    
            except Exception as e:
                print(f"❌ Error loading tool from {py_file}: {e}")
        
        print(f"📊 Total static tools loaded: {len(static_tools)}")
        return static_tools
    
    def _load_dynamic_tools(self):
        """Load dynamic tools from tool registry"""
        try:
            return load_dynamic_tools()
        except Exception as e:
            print(f"❌ Error loading dynamic tools: {e}")
            return []
    
    def _load_all_tools(self):
        """Load all available tools"""
        self.static_tools = self._load_static_tools()
        self.dynamic_tools = self._load_dynamic_tools()
        self.all_tools = self.static_tools + self.dynamic_tools
        
        print(f"📊 Total tools loaded: {len(self.all_tools)} "
              f"(Static: {len(self.static_tools)}, Dynamic: {len(self.dynamic_tools)})")
    
    def find_relevant_tools(self, query: str, max_tools: int = 20) -> List[Tool]:
        """Find tools relevant to the query using semantic similarity"""
        if not self.all_tools:
            return []
        
        # Calculate relevance scores
        tool_scores = []
        query_lower = query.lower()
        
        self._load_all_tools()

        for tool in self.all_tools:
            # Score based on name and description similarity
            name_score = self._calculate_similarity(query_lower, tool.name.lower())
            desc_score = self._calculate_similarity(query_lower, tool.description.lower())
            
            # Weight description higher than name
            total_score = (name_score * 0.3) + (desc_score * 0.7)
            
            # Bonus for exact keyword matches
            keywords = self._extract_keywords(query_lower)
            for keyword in keywords:
                if keyword in tool.name.lower() or keyword in tool.description.lower():
                    total_score += 0.2
            
            tool_scores.append((tool, total_score))
        
        # Sort by relevance and return top tools
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_tools = [tool for tool, score in tool_scores[:max_tools] if score > 0.1]
        
        return relevant_tools
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'get', 'find', 'show', 'tell', 'help', 'please', 'can', 'could', 'would', 'should'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available tools"""
        return self.all_tools
    
    def refresh_tools(self):
        """Refresh tool discovery"""
        self._load_all_tools()

# ✅ Task Planning and Decomposition
class TaskPlanner:
    """Intelligent task planning and decomposition"""
    
    def __init__(self, tool_discovery: ToolDiscovery):
        self.tool_discovery = tool_discovery
    
    def analyze_task(self, user_query: str) -> TaskContext:
        """Analyze user query and create task context"""
        task_id = str(uuid.uuid4())
        
        # Simple task decomposition - can be enhanced with LLM
        sub_tasks = self._decompose_task(user_query)
        
        context = TaskContext(
            task_id=task_id,
            user_query=user_query,
            sub_tasks=sub_tasks,
            completed_steps=[],
            intermediate_results={},
            created_at=datetime.now(),
            status="planning"
        )
        
        return context
    
    def _decompose_task(self, query: str) -> List[str]:
        """Decompose complex query into sub-tasks"""
        # Simple decomposition based on common patterns
        query_lower = query.lower()
        sub_tasks = []
        
        # Check for sequential indicators
        if " and " in query_lower:
            parts = query.split(" and ")
            sub_tasks.extend([part.strip() for part in parts])
        elif " then " in query_lower:
            parts = query.split(" then ")
            sub_tasks.extend([part.strip() for part in parts])
        else:
            # Single task
            sub_tasks.append(query.strip())
        
        return sub_tasks
    
    def select_tools_for_task(self, task: str, context: TaskContext) -> List[Tool]:
        """Select appropriate tools for a specific task"""
        # Find relevant tools
        relevant_tools = self.tool_discovery.find_relevant_tools(task)
        
        # Additional filtering based on context
        if context.intermediate_results:
            # Prefer tools that can work with existing results
            pass
        
        return relevant_tools

# ✅ Enhanced Tool Execution
def execute_tool_with_context(tool: Tool, args: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Execute tool with proper context and error handling"""
    try:
        print(f"\nExecuting tool (inside execute_tool_with_context): {tool.name}")
        # Get function signature
        sig = inspect.signature(tool.func)
        param_names = list(sig.parameters.keys())
        print(f"\nTool parameters: {param_names}")
        
        # Prepare arguments
        if not args:
            print(f"\nNo arguments provided for tool: {tool.name}")
            # No arguments provided
            if len(param_names) == 0:
                result = tool.func()
            else:
                # Check if session_id is expected
                if 'session_id' in param_names:
                    result = tool.func(session_id=session_id)
                else:
                    result = tool.func()
        else:
            # Filter arguments to match function signature
            filtered_args = {}
            for param in param_names:
                if param in args:
                    filtered_args[param] = args[param]
                elif param == 'session_id':
                    filtered_args[param] = session_id
            
            print(f"\nFiltered arguments: {filtered_args}")
            result = tool.func(**filtered_args)
            print(f"\nResult: {result}")
        # Record execution
        session_manager.add_execution_record(session_id, tool.name, args, result, True)
        
        return {
            "success": True,
            "result": result,
            "tool_name": tool.name
        }
        
    except Exception as e:
        error_msg = f"Tool '{tool.name}' error: {str(e)}"
        session_manager.add_execution_record(session_id, tool.name, args, error_msg, False)
        
        return {
            "success": False,
            "error": error_msg,
            "tool_name": tool.name
        }

# ✅ Global Instances
tool_discovery = ToolDiscovery()
task_planner = TaskPlanner(tool_discovery)

# ✅ LangGraph Nodes
def analyze_required_tools(user_query: str, available_tools: List[Tool]) -> Dict[str, Any]:
    """Analyze user query to determine required tools and their sequence using LLM"""
    # Create a prompt for the LLM to analyze the query and identify required tools
    tool_descriptions = []
    for tool in available_tools:
        sig = inspect.signature(tool.func)
        param_info = []
        for param_name, param in sig.parameters.items():
            if param_name not in ['self', 'session_id']:
                param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "string"
                param_info.append(f"{param_name}: {param_type}")
        
        tool_descriptions.append(f"- {tool.name}({', '.join(param_info)}): {tool.description}")

    print("\n Tool Description : ", tool_descriptions)
    analysis_prompt = f"""Analyze the following user query and identify which tools are absolutely necessary to complete the task.
Consider the following:
1. Only select tools that are directly required to fulfill the user's request
2. Consider the dependencies between tools (e.g., if one tool's output is needed for another)
3. Do not select tools that are optional or not directly related to the task
4. Consider the natural sequence of operations needed to complete the task
5. Be strict in tool selection - only include tools that are explicitly needed

User Query: {user_query}

Available Tools:
{chr(10).join(tool_descriptions)}

IMPORTANT: Respond with ONLY the JSON object, no markdown formatting or code blocks.

Provide a JSON response with:
1. required_tools: List of tool names that are absolutely necessary
2. reasoning: Brief explanation of why each tool is required
3. execution_order: Suggested order of tool execution
4. tool_parameters: For each tool, specify what parameters should be passed

Format your response as a JSON object with this structure:
{{
    "required_tools": ["tool1", "tool2"],
    "reasoning": "Explanation of tool selection",
    "execution_order": ["tool1", "tool2"],
    "tool_parameters": {{
        "tool1": {{"param1": "value1"}},
        "tool2": {{"param2": "value2"}}
    }}
}}"""

    try:
        # Use a different LLM instance for tool analysis to avoid interference
        analysis_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=1000,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )
        
        response = analysis_llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Parse the JSON response
        #print(f"Analysis response: {response.content}")
        
        response_content = response.content.strip()
        # Remove markdown code blocks if present
        json_content = response_content
        if response_content.startswith('```json'):
            # Find the JSON content between ```json and ```
            start_marker = '```json'
            end_marker = '```'
            start_idx = response_content.find(start_marker) + len(start_marker)
            end_idx = response_content.find(end_marker, start_idx)
            if end_idx != -1:
                json_content = response_content[start_idx:end_idx].strip()
        elif response_content.startswith('```'):
            # Handle case where it's just ``` without json
            lines = response_content.split('\n')
            if len(lines) > 2 and lines[0].strip() == '```' and lines[-1].strip() == '```':
                json_content = '\n'.join(lines[1:-1]).strip()
                
        try:
            analysis = json.loads(json_content)
            
            # Validate the required structure
            required_keys = ['required_tools', 'reasoning', 'execution_order', 'tool_parameters']
            missing_keys = [key for key in required_keys if key not in analysis]
            if missing_keys:
                print(f"⚠️ Missing keys in analysis: {missing_keys}")
                return fallback_tool_selection(user_query, available_tools)

            # Log the analysis for debugging
            print(f"\n🔍 Tool Analysis for query: {user_query}")
            print(f"Required Tools: {analysis.get('required_tools', [])}")
            print(f"Reasoning: {analysis.get('reasoning', 'No reasoning provided')}")
            print(f"Execution Order: {analysis.get('execution_order', [])}")
            print(f"Tool Parameters: {analysis.get('tool_parameters', {})}")
            
            return analysis
        except json.JSONDecodeError as json_error:
            print(f"❌ Failed to parse LLM response as JSON: {json_content}")
            print(f"JSON Error: {json_error}")
            return fallback_tool_selection(user_query, available_tools)
            
    except Exception as e:
        print(f"❌ Error in tool analysis: {e}")
        return fallback_tool_selection(user_query, available_tools)

def fallback_tool_selection(user_query: str, available_tools: List[Tool]) -> Dict[str, Any]:
    """Fallback method for tool selection using keyword matching"""
    # Extract key actions from the query
    print(f"\n⚠️⚠️Fallback tool selection: {user_query}")
    actions = []
    if "find" in user_query.lower() or "get" in user_query.lower():
        # Split on common conjunctions
        parts = re.split(r'\s*,\s*|\s+and\s+', user_query.lower())
        actions = [part.strip() for part in parts]
    
    # Map common actions to tool names
    action_to_tool = {
        "stock price": "get_stock_price",
        "ceo": "get_company_ceo",
        "financials": "get_company_financials",
        "news": "get_latest_news",
        "weather": "get_weather",
        "summary": "generate_summary",
        "statistics": "get_statistics",
        "translate": "translate_text"
    }
    
    # Find matching tools
    required_tools = []
    tool_parameters = {}
    
    for action in actions:
        for key, tool_name in action_to_tool.items():
            if key in action:
                if tool_name not in required_tools:
                    required_tools.append(tool_name)
                    # Extract company name if present
                    company_match = re.search(r'(?:for|of|about)\s+(\w+)', action)
                    if company_match:
                        tool_parameters[tool_name] = {"symbol": company_match.group(1).upper()}
    
    return {
        "required_tools": required_tools,
        "reasoning": "Fallback tool selection based on keyword matching",
        "execution_order": required_tools,
        "tool_parameters": tool_parameters
    }

def agent_node(state: AgentState):
    """Main agent reasoning node with generic task handling"""
    print(f"\n⭐Entered agent node⭐")
    session_id = state["session_id"]
    user_query = state["user_query"]
    messages = state["messages"]
    intermediate_results = state.get("intermediate_results", {})
    execution_step = state.get("execution_step", 0)
    
    # Safety check for maximum execution steps
    if execution_step > 6:
        return {
            "messages": [AIMessage(content="⚠️ Maximum execution steps reached (currently 6 steps only). Task completed with available results.")],
            "workflow_complete": True,
            "execution_step": execution_step
        }
    
    # Get or create task context
    session = session_manager.get_session(session_id)
    if not session.current_context:
        context = task_planner.analyze_task(user_query)
        session_manager.update_context(session_id, context)
    else:
        context = session.current_context
    
    # Find relevant tools for the entire query
    relevant_tools = tool_discovery.find_relevant_tools(user_query)
    print("\n Relevant tools detected: ", relevant_tools)
    if not relevant_tools:
        return {
            "messages": [AIMessage(content="❌ No relevant tools found for your request. Please try rephrasing your query.")],
            "workflow_complete": True,
            "execution_step": execution_step
        }

    # Analyze required tools and their sequence
    tool_analysis = analyze_required_tools(user_query, relevant_tools)
    required_tools = set(tool_analysis["required_tools"])
    execution_order = tool_analysis["execution_order"]
    tool_parameters = tool_analysis["tool_parameters"]
    
    # Get executed and remaining tools
    executed_tools = set(intermediate_results.keys())
    remaining_tools = [tool for tool in execution_order if tool not in executed_tools]
    print(f"\nAgent node remaining tools: {remaining_tools}")
    print(f"\nAgent node executed tools: {executed_tools}")

    # If all required tools are executed, provide final summary
    if not remaining_tools:
        final_response = "✅ All required tools executed. Final Summary:\n"
        print(f"\nAgent node final response: {final_response}")
        for tool_name in execution_order:
            if tool_name in intermediate_results:
                final_response += f"\n• {tool_name}: {intermediate_results[tool_name]}"
        return {
            "messages": [AIMessage(content=final_response)],
            "intermediate_results": intermediate_results,
            "workflow_complete": True,
            "execution_step": execution_step + 1
        }
    
    # Create system message with dynamic tool information and current state
    tool_descriptions = []
    for tool in relevant_tools:
        if tool.name in required_tools:
            sig = inspect.signature(tool.func)
            param_info = []
            for param_name, param in sig.parameters.items():
                param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "string"
                param_info.append(f"{param_name}: {param_type}")
            
            status = "EXECUTED" if tool.name in executed_tools else "PENDING"
            tool_descriptions.append(f"- {tool.name}({', '.join(param_info)}): {tool.description} [{status}]")
        
    # Add current state information to system message
    state_info = "\nCURRENT STATE:\n"
    if intermediate_results:
        state_info += "Intermediate Results:\n"
        for tool_name in execution_order:
            if tool_name in intermediate_results:
                state_info += f"- {tool_name}: {intermediate_results[tool_name]}\n"
    
    # Add execution guidance
    execution_guidance = f"""
EXECUTION GUIDANCE:
1. You MUST execute ALL required tools in the specified order: {execution_order}
2. After each tool execution, analyze the results and determine the next step
3. Use intermediate results from previous tools as inputs for subsequent tools
4. Continue execution until ALL required tools are executed
5. Only mark the task as complete when ALL required information is gathered

Current Execution Step: {execution_step}
Previous Results: {list(intermediate_results.keys())}
Required Tools: {list(required_tools)}
Executed Tools: {list(executed_tools)}
Next Tool to Execute: {remaining_tools[0] if remaining_tools else "None"}

Remember: Your goal is to COMPLETE THE ENTIRE TASK in the correct sequence.
"""
    
    print(f"\nAgent node state info: {state_info}")
    print("\nprevious tools: ", list(intermediate_results.keys()))
    print("\nprevious tools results: ", list(intermediate_results.values()))
    print("\nRequired tools: ", list(required_tools))
    print("\nExecuted tools: ", list(executed_tools))
    print("\nnext tool to execute: ", remaining_tools[0] if remaining_tools else "None")
    print("\nAvailable tools: ", chr(10).join(tool_descriptions))
    print("\n Execution guidance: ", execution_guidance)
    
    system_msg = SystemMessage(content=f"""You are an intelligent multi-step reasoning agent capable of solving complex tasks.

AVAILABLE TOOLS:
{chr(10).join(tool_descriptions)}

{state_info}

{execution_guidance}

CORE CAPABILITIES:
- Execute tools in the exact sequence specified
- Use tool parameters as provided
- Connect outputs from one tool as inputs to the next
- Provide comprehensive responses

TOOL USAGE RULES:
- Always use the exact parameter names as specified in the tool schema
- Pass parameters as direct key-value pairs
- Execute tools in the specified order
- DO NOT skip any required tools
- DO NOT execute tools that are not in the required list

Current session: {session_id}
User query: {user_query}
""")
    
    # Format tools for Gemini
    formatted_tools = []
    for tool in relevant_tools:
        if tool.name in required_tools and tool.name not in executed_tools:
            sig = inspect.signature(tool.func)
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name not in ['self', 'session_id']:
                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name} for {tool.name}"
                    }
                    
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)
            
            formatted_tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
    
    # Bind tools to LLM
    llm_with_tools = llm.bind(tools=formatted_tools)
    
    # Get response from LLM
    response = llm_with_tools.invoke([system_msg] + messages)
    print(f"\nAgent node response: {response}")
    # Execute tools if needed
    if response.tool_calls:
        tool_results = []
        new_tools_executed = False
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            print(f"\nTool call: {tool_call}")
            print(f"\nTool args: {tool_args}")
            
            # Skip if tool was already executed
            if tool_name in executed_tools:
                print(f"\nEexecuted tools: {executed_tools}")
                print(f"\nTool already executed: {tool_name}")
                continue
                
            # Find the tool
            tool = next((t for t in relevant_tools if t.name == tool_name), None)
            print(f"\nTool found: {tool}")
            
            if tool:
                # Merge tool parameters with provided args
                print(f"\nTool parameters: {tool_parameters}")
                ''' 
                if tool_name in tool_parameters:
                    tool_args.update(tool_parameters[tool_name])
                    print(f"\nTool parameters updated: {tool_args}")
                '''    
                # Execute tool
                execution_result = execute_tool_with_context(tool, tool_args, session_id)
                print(f"\nExecution result: {execution_result}")
                if execution_result["success"]:
                    result_text = f"✅ {tool_name}: {execution_result['result']}"
                    # Store intermediate result for potential use by other tools
                    intermediate_results[tool_name] = execution_result['result']
                    new_tools_executed = True
                else:
                    result_text = f"❌ {execution_result['error']}"
                
                tool_results.append(result_text)
        
        # Create comprehensive response
        final_response = response.content or ""
        print(f"\nComprehensive Final response 1: {final_response}")
        if tool_results:
            final_response += "\n\n" + "\n".join(tool_results)
        print(f"\nComprehensive Final response 2: {final_response}")
        
        # Update executed and remaining tools
        executed_tools = set(intermediate_results.keys())
        remaining_tools = [tool for tool in execution_order if tool not in executed_tools]
        print(f"\nUpdated Remaining tools: {remaining_tools}")
        print(f"\nUpdated Executed tools: {executed_tools}")
        
        # Check if we have completed all required tools
        if not remaining_tools:
            # All required tools executed, provide final summary
            final_response += f"\n\n✅ All required tools executed. Final Summary:\n"
            print(f"\n All required tools executed. Final response: {final_response}")
            for tool_name in execution_order:
                if tool_name in intermediate_results:
                    final_response += f"\n• {tool_name}: {intermediate_results[tool_name]}"
            return {
                "messages": [AIMessage(content=final_response)],
                "intermediate_results": intermediate_results,
                "workflow_complete": True,
                "execution_step": execution_step + 1
            }
        elif new_tools_executed:
            # Continue execution if we made progress and have more tools to run
            final_response += f"\n\n🔄 Progress made. Next tool to execute: {remaining_tools[0]}"
            print(f"\n More tools to run. Final response: {final_response}")
            print(f"\nIntermediate results: {intermediate_results}")
            print(f"\nNext Execution step: {execution_step + 1}")
            return {
                "messages": [AIMessage(content=final_response)],
                "intermediate_results": intermediate_results,
                "workflow_complete": False,
                "execution_step": execution_step + 1
            }
        else:
            # No new tools executed, but we still have remaining tools
            final_response += f"\n\n⚠️ No new tools executed. Next tool to execute: {remaining_tools[0]}"
            print(f"\n No new tools executed. But we still have remaining tools. Final response: {final_response}")
            return {
                "messages": [AIMessage(content=final_response)],
                "intermediate_results": intermediate_results,
                "workflow_complete": False,
                "execution_step": execution_step + 1
            }
    
    # Handle case when no tools were called
    if remaining_tools:
        # Still have tools to execute, continue
        final_response = response.content + f"\n\n🔄 Continuing with next tool: {remaining_tools[0]}"
        return {
            "messages": [AIMessage(content=final_response)],
            "intermediate_results": intermediate_results,
            "workflow_complete": False,
            "execution_step": execution_step + 1
        }
    else:
        # All tools done, provide final summary
        final_response = response.content + f"\n\n✅ Task completed. Summary of results:\n"
        for tool_name in execution_order:
            if tool_name in intermediate_results:
                final_response += f"\n• {tool_name}: {intermediate_results[tool_name]}"
        return {
            "messages": [AIMessage(content=final_response)],
            "intermediate_results": intermediate_results,
            "workflow_complete": True,
            "execution_step": execution_step + 1
        }

def check_confirmations_node(state: AgentState):
    """Check for pending confirmations"""
    print(f"\n⭐Entered check_confirmations_node⭐")
    session_id = state["session_id"]
    
    # Get pending confirmations
    pending_confirmations = session_manager.get_pending_confirmations(session_id)
    print(f"\nPending confirmations: {pending_confirmations}")
    if pending_confirmations:
        print(f"\nPending confirmations found. Returning to agent")
        return {
            "pending_confirmations": [asdict(req) for req in pending_confirmations],
            "workflow_complete": False
        }
    print(f"\nNo pending confirmations found. Return not set yet.")
    return {"pending_confirmations": []}

def should_continue(state: AgentState) -> Literal["agent", "check_confirmations", END]:
    """Decide whether to continue execution or end"""
    print(f"\n⭐Entered should_continue⭐")
    print(f"\nState: {state}")
    
    if state.get("workflow_complete", False):
        print(f"\n✔️Workflow complete. Returning to check_confirmations")
        return "check_confirmations"
    
    # Get the current state
    session_id = state["session_id"]
    intermediate_results = state.get("intermediate_results", {})
    execution_step = state.get("execution_step", 0)
    
    # Get session and context
    session = session_manager.get_session(session_id)
    try:
        print(f"\nSession: {session}")
    except Exception as e:
        print(f"\nError in should_continue getting session: {e}")
        
    if not session or not session.current_context:
        return "agent"
    
    # Get tool information from state instead of recalculating
    executed_tools = set(intermediate_results.keys())
    
    # Get required tools from task context if available
    required_tools = set()
    if session.current_context and hasattr(session.current_context, 'sub_tasks'):
        # Find relevant tools for the query
        relevant_tools = tool_discovery.find_relevant_tools(session.current_context.user_query)
        # Get tool analysis from agent node
        tool_analysis = analyze_required_tools(session.current_context.user_query, relevant_tools)
        required_tools = set(tool_analysis.get("required_tools", []))
    
    # Calculate remaining tools
    remaining_tools = required_tools - executed_tools
    print(f"\nInside should_continue: Remaining tools: {remaining_tools}")
    print(f"\nInside should_continue: Executed tools: {executed_tools}")
    print(f"\nInside should_continue: Required tools: {required_tools}")
    print(f"\nInside should_continue: Execution step: {execution_step}")
    
    # Continue if we have remaining tools and haven't exceeded max steps
    if len(remaining_tools) > 0 and execution_step < 6:
        return "agent"
    
    # Mark as complete if all tools are executed or max steps reached
    if len(remaining_tools) == 0 or execution_step >= 5:
        state["workflow_complete"] = True
        return "check_confirmations"
    
    return "agent"

def confirmation_routing(state: AgentState) -> Literal["agent", END]:
    """Route based on confirmations"""
    if state.get("pending_confirmations"):
        return END  # Exit to handle confirmations in UI
    return END

# ✅ Build LangGraph
def create_generic_reasoning_graph():
    """Create the generic reasoning workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("check_confirmations", check_confirmations_node)
    
    # Add edges
    workflow.set_entry_point("agent")
    
    # Modified edge logic to support continuous execution
    workflow.add_conditional_edges("agent", should_continue,
        {
            "agent": "agent",  # Continue execution
            "check_confirmations": "check_confirmations",
            END: END
        }
    )
    
    workflow.add_conditional_edges("check_confirmations", confirmation_routing,
        {"agent": "agent", END: END}
    )
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory, interrupt_before=["check_confirmations"])

# ✅ Main Agent Class
class GenericReasoningAgent:
    """Generic multi-step reasoning agent"""
    
    def __init__(self):
        self.graph = create_generic_reasoning_graph()
        self.tool_discovery = tool_discovery
        self.task_planner = task_planner
    
    def create_session(self, session_id: Optional[str] = None, user_id: str = None) -> str:
        """Create a new user session"""
        return session_manager.create_session(session_id, user_id)
    
    def run(self, user_input: str, session_id: str, thread_id: str = None):
        """Run the agent with session context"""
        if not thread_id:
            thread_id = session_id
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
            "user_query": user_input,
            "task_context": None,
            "pending_confirmations": [],
            "workflow_complete": False,
            "intermediate_results": {},
            "execution_step": 0
        }
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            return result
        except Exception as e:
            print(f"❌ Agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Agent execution failed: {str(e)}"}
    
    def get_pending_confirmations(self, session_id: str) -> List[ConfirmationRequest]:
        """Get pending confirmations for session"""
        return session_manager.get_pending_confirmations(session_id)
    
    def handle_confirmation(self, session_id: str, confirmation_id: str, approved: bool) -> Dict[str, Any]:
        """Handle confirmation response"""
        try:
            if approved:
                session_manager.update_confirmation_status(session_id, confirmation_id, "approved")
                return {
                    "success": True,
                    "message": "✅ Action approved and will be executed",
                    "confirmation_id": confirmation_id
                }
            else:
                session_manager.update_confirmation_status(session_id, confirmation_id, "rejected")
                return {
                    "success": True,
                    "message": "❌ Action cancelled by user",
                    "confirmation_id": confirmation_id
                }
        except Exception as e:
            return {"success": False, "message": f"Error handling confirmation: {str(e)}"}
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = session_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "current_task": asdict(session.current_context) if session.current_context else None,
                "pending_confirmations_count": len([req for req in session.pending_confirmations if req.status == "pending"]),
                "execution_history_count": len(session.execution_history)
            }
        return None
    
    def cleanup_session(self, session_id: str):
        """Clean up a session"""
        session_manager.delete_session(session_id)
    
    def refresh_tools(self):
        """Refresh available tools"""
        self.tool_discovery.refresh_tools()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tool_discovery.get_all_tools()]

# ✅ Create Global Agent Instance
reasoning_agent = GenericReasoningAgent()

# ✅ Convenience Functions for External Usage
def create_new_session(session_id: Optional[str] = None, user_id: str = None) -> str:
    """Create a new session - convenience function"""
    return reasoning_agent.create_session(session_id, user_id)

def run_query(user_input: str, session_id: str) -> Dict[str, Any]:
    """Run a query - convenience function"""
    return reasoning_agent.run(user_input, session_id)

def get_pending_confirmations(session_id: str) -> List[ConfirmationRequest]:
    """Get pending confirmations - convenience function"""
    return reasoning_agent.get_pending_confirmations(session_id)

def handle_confirmation(session_id: str, confirmation_id: str, approved: bool) -> Dict[str, Any]:
    """Handle confirmation - convenience function"""
    return reasoning_agent.handle_confirmation(session_id, confirmation_id, approved)

def get_available_tools() -> List[str]:
    """Get available tools - convenience function"""
    return reasoning_agent.get_available_tools()

def refresh_tools():
    """Refresh tools - convenience function"""
    reasoning_agent.refresh_tools()

# ✅ CLI Testing Function
def test_agent():
    """Test the generic reasoning agent"""
    # Print header with decorative elements
    print("\n" + "="*80)
    print("🤖 GENERIC MULTI-STEP REASONING AGENT TEST".center(80))
    print("="*80 + "\n")

    # Print available tools section
    print("📋 AVAILABLE TOOLS:")
    print("-"*40)
    tools = get_available_tools()
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool}")
    print("-"*40 + "\n")
    
    # Create session
    session_id = create_new_session()
    print("🔑 SESSION INFORMATION:")
    print("-"*40)
    print(f"Session ID: {session_id}")
    print("-"*40 + "\n")
    
    # Test queries
    # test_queries = [
    #     "Find stock price of APPL, find top 5 news articles about AAPL, and summarize them "    #✅
    #     "Find stock price of APPL, find ceo name, get company financials"    #✅
    #     "Find weather in Nagpur, India"    #✅
    #     "Find stock price of APPL, find top 5 news articles about AAPL, find weather in Nagpur, India"    #✅
    #     "Find top 5 news articles about Elon Musk, analyze sentiment and provide summary"  #✅
    # ]
    test_queries = [  
        "Find top 5 news articles about Elon Musk, provide summary and count the frequency of words from summary" 
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"🔍 TEST QUERY #{i}:")
        print(f"Query: {query}")
        print("-"*40)
        
        # Run query and format result
        result = run_query(query, session_id)
        
        # Format messages if present
        if "messages" in result:
            print("\n📝 AGENT RESPONSES:")
            print("-"*40)
            for msg in result["messages"]:
                print(f"\n{msg.content}")
        
        # Format intermediate results if present
        if "intermediate_results" in result:
            print("\n📊 INTERMEDIATE RESULTS:")
            print("-"*40)
            for tool_name, result_value in result["intermediate_results"].items():
                print(f"\n• {tool_name}:")
                print(f"  {result_value}")
        
        # Format any errors if present
        if "error" in result:
            print("\n❌ ERROR:")
            print("-"*40)
            print(result["error"])
        
        print(f"\n{'='*80}")
    
    # Check and display session info
    print("\n📋 SESSION SUMMARY:")
    print("-"*40)
    info = reasoning_agent.get_session_info(session_id)
    if info:
        print(f"Session ID: {info['session_id']}")
        print(f"Created At: {info['created_at']}")
        print(f"Last Activity: {info['last_activity']}")
        print(f"Pending Confirmations: {info['pending_confirmations_count']}")
        print(f"Execution History Entries: {info['execution_history_count']}")
    print("-"*40 + "\n")

if __name__ == "__main__":
    test_agent()