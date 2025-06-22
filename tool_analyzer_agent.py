import os
import glob
import uuid
from typing import Dict, List, Literal, TypedDict, Annotated, Optional, Any, Union
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import agents
from agents.tool_user_agent import GenericReasoningAgent
from agents.tool_generator_agent import CodeGeneratorAgent

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=500,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True
)

# State definition for the multi-agent system
class AgentState(TypedDict):
    """State schema shared between all agents."""
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    session_id: str
    user_query: str
    required_tools: List[str]
    missing_tools: List[str]
    generated_tools: List[str]
    workflow_complete: bool
    next_agent: Optional[Literal["tool_creator", "tool_user", END]]

# Tool Selection Expert Node
def tool_selection_expert(state: AgentState) -> AgentState:
    """Analyzes user query and determines required tools."""
    print("\n🔍 Tool Selection Expert analyzing query...")
    
    # Get user query
    query = state["user_query"]

    # Dynamically list all available tools (static and dynamic)
    def list_all_tools():
        tool_files = []
        for tool_path in glob.glob("tools/*.py"):
            if "__" not in tool_path:
                tool_files.append(os.path.basename(tool_path))
        for tool_path in glob.glob("dynamic_tools/*.py"):
            if "__" not in tool_path:
                tool_files.append(os.path.basename(tool_path))
        return tool_files

    available_tools = list_all_tools()
    available_tools_str = "\n".join(f"- {tool}" for tool in available_tools)

    # System message for tool analysis (tool-agnostic)
    system_msg = SystemMessage(content=f"""You are a Tool Selection Expert.
    Your job is to analyze user queries and determine what tools are needed to fulfill them.
    
    Available Tools:\n{available_tools_str}
    
    Analyze the query and list ALL tools needed to fulfill it completely.
    Do NOT list tools that don't exist or aren't mentioned above.
    List tools in a simple format, one per line, with just the filename (e.g. 'get_stock_price.py').
    Do not use numbers, backticks, or any other formatting.
    If all Tools are not available to fulfill the request completely give response as "New Tool Needed"
    If even a single Tool is missing to solve request completely give response as "New Tool Needed"
    """)
    
    # Get tool analysis from LLM
    messages = [
        system_msg,
        HumanMessage(content=f"Analyze this query and list all required tools: {query}")
    ]
    
    response = llm.invoke(messages)
    print("🔍🔍🔍tool_selection_expert response := ", response)
    
    # Get existing tools from both directories
    existing_tools = set(available_tools)
    
    # Check if LLM response indicates new tool is needed
    response_content = response.content.strip().lower()
    if "new tool needed" in response_content:
        print("🛠️⚒️⚒️ LLM indicates new tool is needed")
        required_tools = []
    else:
        # Extract required tools from LLM response
        required_tools = []
        for line in response.content.split("\n"):
            if ".py" in line:
                # Clean up the tool name by removing numbers, backticks, and extra spaces
                tool_name = line.split(".py")[0].strip()
                tool_name = ''.join(c for c in tool_name if not c.isdigit() and c != '`' and c != '.')
                tool_name = tool_name.strip() + ".py"
                required_tools.append(tool_name)
    
    print(f"📋 Required tools: {required_tools}")
    # Check if required_tools is empty
    if not required_tools:
        print("❌ No required tools identified")
        next_agent = "tool_creator"
    else:
        print("✅ All required tools are available")
        next_agent = "tool_user"
    
    # Add analysis result to messages
    if "new tool needed" in response_content:
        analysis_msg = AIMessage(content=f"Tool Analysis: New tool creation required for query: {query}")
    else:
        analysis_msg = AIMessage(content=f"Tool Analysis:\nRequired tools: {required_tools}")
    
    return {
        "messages": state["messages"] + [analysis_msg],
        "session_id": state["session_id"],
        "user_query": query,
        "required_tools": required_tools,
        "missing_tools": [],
        "generated_tools": [],
        "workflow_complete": False,
        "next_agent": next_agent
    }

# Tool Creator Node
def tool_creator(state: AgentState) -> AgentState:
    """Generates tools using code_generator_agent based on user query."""
    print("\n============== ⭐🛠️ Tool Creator generating tools... =======================")
    
    # Initialize code generator agent
    code_gen = CodeGeneratorAgent()
    generated_tools = []
    generation_messages = []
    
    # Generate tool based on user query
    print(f"\n📝 Generating tool for query: {state['user_query']}")
    result = code_gen.create_tool(
        f"Create a tool that can handle this query: {state['user_query']}"
    )
    print('🌟🌟🌟Result from Tool Creator = ', result)
    if result["status"] == "success":
        tool_name = result.get('function_name', "custom_tool.py")
        generated_tools.append(tool_name)
        generation_messages.append(AIMessage(content=f"✅ Generated tool: {tool_name}"))
        print(f"✅ Generated {tool_name} successfully")
    else:
        generation_messages.append(AIMessage(content=f"❌ Failed to generate tool: {result['message']}"))
        print(f"❌ Failed to generate tool: {result['message']}")
    
    return {
        **state,
        "messages": state["messages"] + generation_messages,
        "generated_tools": generated_tools,
        "workflow_complete": False,
        "next_agent": "tool_user"
    }

# Tool User Node (generalized from stock_monitor)
def tool_user(state: AgentState) -> AgentState:
    """Executes the query using the generic tool user agent."""
    print("\n==================⭐⚒️ Tool User executing query... =================")
    # Initialize generic reasoning agent
    user_agent = GenericReasoningAgent()
    
    # Create a session
    session_id = state.get("session_id")
    if not session_id:
        session_id = user_agent.create_session()
        print(f"📝 Created new session: {session_id}")
        
    # Build new user_query message
    tools_list_req = "', '".join(tool.replace(".py", "") for tool in state['required_tools'])
    tools_list_gen = "', '".join(tool.replace(".py", "") for tool in state['generated_tools'])
    if tools_list_gen:
        use_these_tools = tools_list_gen
    else:
        use_these_tools = tools_list_req
    user_query = f"{state['user_query']}. Use all these tools : '{use_these_tools}'"
    
    # Execute query
    result = user_agent.run(user_query, session_id)
    print("🔍🔍🔍(Inside multi_agent_system) result from tool_user_agent := ", result)
    # Add result to messages
    if isinstance(result, dict) and "messages" in result:
        messages = state["messages"] + result["messages"]
    else:
        messages = state["messages"] + [AIMessage(content=str(result))]
    print("🔍🔍🔍(Inside multi_agent_system) messages receieved from tool_user_agent:= ", messages)
    return {
        **state,
        "messages": messages,
        "session_id": session_id,
        "workflow_complete": True,
        "next_agent": END
    }

def route_next(state: AgentState) -> Literal["tool_creator", "tool_user", END]:
    """Routes to the next agent based on state."""
    return state["next_agent"] or END

# Create the multi-agent workflow
def create_multi_agent_workflow() -> StateGraph:
    """Creates the LangGraph workflow for the multi-agent system."""
    workflow = StateGraph(AgentState)
    workflow.add_node("tool_selection_expert", tool_selection_expert)
    workflow.add_node("tool_creator", tool_creator)
    workflow.add_node("tool_user", tool_user)
    workflow.set_entry_point("tool_selection_expert")
    workflow.add_conditional_edges(
        "tool_selection_expert",
        route_next,
        {
            "tool_creator": "tool_creator",
            "tool_user": "tool_user"
        }
    )
    workflow.add_conditional_edges(
        "tool_creator",
        route_next,
        {
            "tool_user": "tool_user",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "tool_user",
        route_next,
        {END: END}
    )
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

class MultiAgentSystem:
    """Main class for the multi-agent system."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        # Create directories if they don't exist
        if not os.path.exists("dynamic_tools"):
            os.makedirs("dynamic_tools")
            with open("dynamic_tools/__init__.py", "w") as f:
                f.write("# Dynamic tools package\n")
        
        self.graph = create_multi_agent_workflow()
    
    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run a query through the multi-agent system."""
        try:
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "session_id": session_id or str(uuid.uuid4()),
                "user_query": query,
                "required_tools": [],
                "missing_tools": [],
                "generated_tools": [],
                "workflow_complete": False,
                "next_agent": None
            }
            print("⚒️⚒️⚒️initial_state", initial_state)
            # Configure checkpointer with required keys
            config = {
                "configurable": {
                    "thread_id": initial_state["session_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": None
                }
            }
            # Run the workflow with configured checkpointer
            result = self.graph.invoke(initial_state, config=config)
            res = {
                "status": "success",
                "messages": result["messages"],
                "session_id": result["session_id"]
            }
            print("🚀🚀🚀res", res)
            return res
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }

# Create global instance
multi_agent_system = MultiAgentSystem()

def run_query(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run a query through the multi-agent system."""
    return multi_agent_system.run(query, session_id)

if __name__ == "__main__":
    # Test the system
    print("🤖 Testing Multi-Agent System")
    print("=" * 60)
    session_id = str(uuid.uuid4())
    # test_queries = [
    #     "Get the current price of Apple stock", #✅
    #     "Monitor Tesla stock and send me an email at himunagapure114@gmail.com, when it goes above $250",
    #     "Fetch top 5 current news headlines for a stock INTC" #✅
    #      "Get current price of stock AAPL and get me the top 10 news on it" #✅
    #     "Find stock price of APPL, find top 5 news articles about AAPL, and summarize them "    #✅
    #     "Find stock price of APPL, find ceo name, get company financials"    #✅
    #     "Find weather in Nagpur, India"    #✅
    #     "Find stock price of APPL, find top 5 news articles about AAPL, find weather in Nagpur, India"    #✅
    #     "Find top 5 news articles about Elon Musk, analyze sentiment and provide summary"  #✅
    #   "Find top 5 news articles about Elon Musk, provide summary and count the frequency of word 'Elon' from summary" ❌
    #   ""Find sum of 7 and 10, find the factorial of the output sum and divide it by 2"" #✅
    # ]
    test_queries = [
        "Find top 5 news articles about Elon Musk, provide summary and count the frequency of word 'Elon' from summary" 
    ]
    
    for query in test_queries:
        print(f"\n🚀 Testing query: {query}")
        print("-" * 50)
        
        result = run_query(query, session_id=session_id)
        
        if result["status"] == "success":
            print("\n✅ Query executed successfully")
            for msg in result["messages"]:
                if hasattr(msg, 'content'):
                    print(f"📋 {msg.content}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("=" * 60) 