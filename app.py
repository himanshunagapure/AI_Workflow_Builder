import streamlit as st
import uuid
import re
from tool_analyzer_agent import run_query

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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0.5em;'>AI Workflow Builder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Build and run complex AI workflows with a single query.</p>", unsafe_allow_html=True)

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

with st.form(key="workflow_form", clear_on_submit=False):
    user_query = st.text_input("Enter your workflow/task:", "", key="query_input", help="Describe what you want the AI to do.")
    submit = st.form_submit_button("Submit")

if submit and user_query.strip():
    st.session_state['last_query'] = user_query
    st.session_state['last_result'] = None
    st.session_state['last_error'] = None
    with st.spinner("Processing your workflow..."):
        try:
            result = run_query(user_query, session_id=st.session_state['session_id'])
            st.session_state['last_result'] = result
        except Exception as e:
            st.session_state['last_error'] = str(e)

if st.session_state.get('last_error'):
    st.error(f"Error: {st.session_state['last_error']}")

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