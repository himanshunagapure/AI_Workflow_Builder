import streamlit as st
import uuid
import re
from tool_analyzer_agent import run_query
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="AI Workflow Builder", page_icon="🤖", layout="centered")

st.markdown("""
<style>
body, .stApp { background: #f8f9fa; }
.block-container { max-width: 600px; margin: auto; }
.stTextInput>div>div>input { text-align: center; }
.stButton>button { width: 100%; }
.final-result {
    background: #e6f7ee;
    border-left: 5px solid #2ecc71;
    padding: 1em 1.5em;
    margin: 1.5em 0 1em 0;
    border-radius: 8px;
    font-size: 1.08em;
}
.step-header { color: #2d7be5; font-weight: 600; margin-top: 1em; }
.example-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    color: white;
    padding: 10px 15px;
    margin: 5px;
    cursor: pointer;
    transition: all 0.3s;
}
.info-card {
    background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
    border-radius: 10px;
    padding: 15px;
    border-left: 4px solid #4c6ef5;
    margin: 10px 0;
}
.workflow-analysis {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border: 2px solid #ffc107;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
}
.workflow-step {
    background: #ffffff;
    border-left: 4px solid #ff9800;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0.5em;'>AI Workflow Builder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Build and run complex AI workflows with a single query.</p>", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'pending_query' not in st.session_state:
    st.session_state['pending_query'] = None

if 'workflow_analysis' not in st.session_state:
    st.session_state['workflow_analysis'] = None

# Function to analyze workflow with LLM
def analyze_workflow_with_llm(user_query, result):
    """Analyze the workflow results and generate a comprehensive step-by-step explanation"""
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Extract information from the result
        messages = result.get('messages', [])
        session_id = result.get('session_id', 'Unknown')
        
        # Build a comprehensive analysis prompt
        analysis_prompt = f"""
You are an expert AI workflow analyst. Your task is to analyze a user's query and the AI system's execution results to create a comprehensive, step-by-step workflow that any beginner can follow.

USER QUERY: {user_query}

EXECUTION RESULTS:
Session ID: {session_id}

Please analyze the following execution details and create a clear, educational workflow:

1. **Extract all the tools and steps used** from the execution
2. **Identify the reasoning path** the AI took
3. **Create a beginner-friendly workflow** that explains:
   - What tools were used and why
   - What data was processed at each step
   - How the final result was achieved
   - What a user would need to replicate this workflow

EXECUTION MESSAGES:
"""
        
        # Add all messages to the prompt
        for i, msg in enumerate(messages):
            if hasattr(msg, 'content') and msg.content:
                analysis_prompt += f"\nStep {i+1}:\n{msg.content}\n"
        
        analysis_prompt += """

Please provide your analysis in the following format:

## 🔍 Workflow Analysis

### 📋 Original Query
[Brief description of what the user wanted]

### 🛠️ Tools Used
[List and explain each tool that was utilized]

### 📊 Step-by-Step Workflow
[Numbered steps with clear explanations]

### 🎯 Key Insights
[What made this workflow successful]

### 📚 Learning Points
[What a beginner can learn from this workflow]

### 🔄 How to Replicate
[Clear instructions for someone to recreate this workflow]

Focus on making this educational and transparent. Even if there were errors in the original execution, complete the workflow analysis showing what the intended flow should be.
"""
        
        # Get LLM response
        response = llm.invoke(analysis_prompt)
        
        return response.content
        
    except Exception as e:
        return f"Error generating workflow analysis: {str(e)}"

# Example Queries Function
def show_examples():
    """Show example queries"""
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    examples = [
        {
            "title": "📰 Stock News Headlines",
            "query": "Fetch top 5 current news headlines for a stock INTC"
        },
        {
            "title": "📊 Stock Price & News",
            "query": "Get current price of stock AAPL and get me the top 10 news on it"
        },
        {
            "title": "🏢 Company Analysis",
            "query": "Find stock price of APPL, find ceo name, get company financials"
        },
        {
            "title": "🌤️ Weather Check",
            "query": "Find weather in Nagpur, India"
        },
        {
            "title": "🔗 Multi-Task Workflow",
            "query": "Find stock price of APPL, find top 5 news articles about AAPL, find weather in Nagpur, India"
        },
        {
            "title": "📝 News Analysis",
            "query": "Find top 5 news articles about Elon Musk, analyze sentiment and provide summary"
        },
        {
            "title": "🍎 Simple Stock Price",
            "query": "Get the current price of Apple stock"
        },
        {
            "title": "🧮 Math Operations",
            "query": "Find sum of 7 and 10, find the factorial of the output sum and divide it by 2"
        }
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(
                f"{example['title']}", 
                key=f"example_{i}",
                help=example['query'],
                use_container_width=True
            ):
                # Set the pending query and run it directly
                st.session_state['pending_query'] = example['query']
                st.rerun()

# Check if there's a pending query from example button
if st.session_state.get('pending_query'):
    query_to_run = st.session_state['pending_query']
    st.session_state['pending_query'] = None  # Clear it
    
    # Run the query directly
    st.session_state['last_query'] = query_to_run
    st.session_state['last_result'] = None
    st.session_state['last_error'] = None
    st.session_state['workflow_analysis'] = None
    
    with st.spinner("Processing your workflow..."):
        try:
            result = run_query(query_to_run, session_id=st.session_state['session_id'])
            st.session_state['last_result'] = result
            
            # Generate workflow analysis
            with st.spinner("Analyzing workflow..."):
                analysis = analyze_workflow_with_llm(query_to_run, result)
                st.session_state['workflow_analysis'] = analysis
                
        except Exception as e:
            st.session_state['last_error'] = str(e)

# Main form for manual input
with st.form(key="workflow_form", clear_on_submit=False):
    user_query = st.text_input("Enter your workflow/task:", key="query_input", help="Describe what you want the AI to do.")
    submit = st.form_submit_button("Submit")

# Handle manual form submission
if submit and user_query and user_query.strip():
    st.session_state['last_query'] = user_query
    st.session_state['last_result'] = None
    st.session_state['last_error'] = None
    st.session_state['workflow_analysis'] = None
    
    with st.spinner("Processing your workflow..."):
        try:
            result = run_query(user_query, session_id=st.session_state['session_id'])
            st.session_state['last_result'] = result
            
            # Generate workflow analysis
            with st.spinner("Analyzing workflow..."):
                analysis = analyze_workflow_with_llm(user_query, result)
                st.session_state['workflow_analysis'] = analysis
                
        except Exception as e:
            st.session_state['last_error'] = str(e)

if st.session_state.get('last_error'):
    st.error(f"Error: {st.session_state['last_error']}")

# Display workflow analysis if available
if st.session_state.get('workflow_analysis'):
    st.markdown("---")
    st.markdown('<div class="workflow-analysis">', unsafe_allow_html=True)
    st.markdown("### 🔍 AI Workflow Analysis")
    st.markdown("*Comprehensive breakdown of how the AI solved your query*")
    
    # Display the analysis with proper formatting
    analysis_content = st.session_state['workflow_analysis']
    
    # Split the analysis into sections for better display
    sections = analysis_content.split('\n\n')
    
    for section in sections:
        if section.strip():
            if section.startswith('##'):
                st.markdown(f"**{section[3:]}**")
            elif section.startswith('###'):
                st.markdown(f"**{section[4:]}**")
            elif section.startswith('**'):
                st.markdown(section)
            else:
                st.markdown(section)
    
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get('last_result'):
    result = st.session_state['last_result']
    if result.get('status') == 'success':
        st.markdown("---")
        st.markdown("### Workflow Results")
        messages = result.get('messages', [])
        final_result_found = False
        for i, msg in enumerate(messages):
            if hasattr(msg, 'content'):
                content = msg.content
                # Check for the final result phrase
                if "✅ All required tools executed." in content:
                    final_result_found = True
                    # Extract only the text after the phrase
                    after_phrase = content.split("✅ All required tools executed.", 1)[-1].strip()
                    if after_phrase:
                        st.markdown("<div class='step-header'>Final Result</div>", unsafe_allow_html=True)
                        # Display the full content, preserving all text
                        st.markdown(f"<div class='final-result'>{after_phrase}</div>", unsafe_allow_html=True)
                else:
                    # Show intermediate steps only if not the final result
                    if not final_result_found:
                        st.markdown(f"<div class='step-header'>Step {i+1}</div>", unsafe_allow_html=True)
                        st.markdown(msg.content)
        st.markdown(f"<div style='text-align:center; color:#888; font-size:0.9em;'>Session ID: {result.get('session_id')}</div>", unsafe_allow_html=True)
    else:
        st.error(f"Error: {result.get('error', 'Unknown error')}")

# Show examples after the form
show_examples() 