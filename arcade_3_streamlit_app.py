"""
Streamlit UI for Arcade AI Agent with LangGraph and Supabase Authentication.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import supabase
from supabase.client import Client
import uuid
import asyncio
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

# Import our agent
from arcade_3_agent_with_memory import build_graph

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Get other environment variables
arcade_api_key = os.environ.get("ARCADE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_choice = os.environ.get("MODEL_CHOICE")
database_url = os.environ.get("DATABASE_URL")

# Streamlit page configuration
st.set_page_config(
    page_title="Arcade AI Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .stMarkdown h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #1f77b4;
    }
    .auth-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def sign_up(email, password, full_name):
    try:
        response = supabase_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })
        if response and response.user:
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing up: {str(e)}")
        return None

def sign_in(email, password):
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response and response.user:
            # Store user info directly in session state
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing in: {str(e)}")
        return None

def sign_out():
    try:
        supabase_client.auth.sign_out()
        # Clear authentication-related session state
        st.session_state.authenticated = False
        st.session_state.user = None
        # Clear conversation history
        st.session_state.messages = []
        # Generate new thread ID for next session
        st.session_state.thread_id = str(uuid.uuid4())
        # Set a flag to trigger rerun on next render
        st.session_state.logout_requested = True
    except Exception as e:
        st.error(f"Error signing out: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Check for logout flag and clear it after processing
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

# Custom writer function for streaming
class StreamlitWriter:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.content = ""
    
    def __call__(self, text):
        # Add proper formatting for authorization messages
        if "🔐 Authorization required" in text:
            # Add newlines around authorization messages
            if not self.content.endswith("\n"):
                self.content += "\n"
            self.content += text
            if not text.endswith("\n"):
                self.content += "\n"
        elif "Visit the following URL to authorize:" in text:
            # Add newline before and after URL instruction
            if not self.content.endswith("\n"):
                self.content += "\n"
            self.content += text + "\n"
        elif "Waiting for authorization..." in text:
            # Add newlines around waiting message
            if not self.content.endswith("\n"):
                self.content += "\n"
            self.content += text + "\n\n"
        elif text.startswith("You have the following emails") or text.startswith("From "):
            # Add newline before email content
            if not self.content.endswith("\n"):
                self.content += "\n"
            self.content += text
        else:
            self.content += text
        
        self.placeholder.markdown(self.content)

async def stream_agent_response(user_input: str, graph, config: dict, placeholder):
    """Stream agent response with proper handling."""
    
    # Build conversation history
    messages = []
    
    # Add conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Add current user message
    messages.append(HumanMessage(content=user_input))
    
    # Prepare input
    inputs = {
        "messages": messages
    }
    
    writer = StreamlitWriter(placeholder)
    full_response = ""
    
    try:
        # Stream the response
        async for chunk in graph.astream(inputs, config=config, stream_mode="custom"):
            if isinstance(chunk, str):
                writer(chunk)
                full_response += chunk
            elif isinstance(chunk, bytes):
                decoded = chunk.decode('utf-8')
                writer(decoded)
                full_response += decoded
        
        return full_response
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        writer(error_msg)
        return error_msg

async def run_agent_interaction(user_input: str, user_email: str, thread_id: str):
    """Run the agent interaction with proper setup."""
    
    # Set up configuration
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_email
        },
        "recursion_limit": 100
    }
    
    # Create placeholder for streaming
    response_placeholder = st.empty()
    
    # Use context managers for store and checkpointer
    async with (
        AsyncPostgresStore.from_conn_string(database_url) as store,
        AsyncPostgresSaver.from_conn_string(database_url) as checkpointer,
    ):
        # Build the graph
        graph = build_graph(checkpointer, store)
        
        # Stream the response
        response = await stream_agent_response(
            user_input, 
            graph, 
            config, 
            response_placeholder
        )
        
        return response

# Sidebar for authentication
with st.sidebar:
    st.title("🎮 Arcade AI Agent")
    
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_button = st.button("Login")
            
            if login_button:
                if login_email and login_password:
                    sign_in(login_email, login_password)
                else:
                    st.warning("Please enter both email and password.")
        
        with tab2:
            st.subheader("Sign Up")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_name = st.text_input("Full Name", key="signup_name")
            signup_button = st.button("Sign Up")
            
            if signup_button:
                if signup_email and signup_password and signup_name:
                    response = sign_up(signup_email, signup_password, signup_name)
                    if response and response.user:
                        st.success("Sign up successful! Please check your email to confirm your account.")
                    else:
                        st.error("Sign up failed. Please try again.")
                else:
                    st.warning("Please fill in all fields.")
    else:
        user = st.session_state.user
        if user:
            st.success(f"Logged in as: {user.email}")
            st.button("Logout", on_click=sign_out)
            
            # Display session information
            st.divider()
            st.subheader("Session Info")
            st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
            
            # Clear conversation button
            if st.button("🔄 New Conversation"):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
            
            # Instructions
            st.divider()
            st.markdown("""
            ### 💡 How to Use
            
            **Email Commands:**
            - "What emails do I have?"
            - "Show me emails from today"
            - "Search for emails about meetings"
            
            **Memory Features:**
            - Include "remember" in your message to store information
            - The agent will recall relevant memories automatically
            
            **Web Scraping:**
            - "Scrape this URL: [website]"
            - "Get information from [website]"
            
            The agent has access to Gmail and web scraping tools.
            """)

# Main chat interface
if st.session_state.authenticated and st.session_state.user:
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎮 Arcade AI Agent Chat")
    st.markdown("AI assistant with email access, web scraping, and memory capabilities")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display all messages from conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (naturally appears at bottom)
    if prompt := st.chat_input("Ask me anything..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            try:
                # Handle Windows event loop
                if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                
                # Run the async agent interaction
                response = asyncio.run(
                    run_agent_interaction(
                        prompt,
                        st.session_state.user.email,
                        st.session_state.thread_id
                    )
                )
                
                # Add both messages to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                # Still add the user message and error to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    # Welcome screen for non-authenticated users
    st.title("Welcome to Arcade AI Agent")
    st.write("Please login or sign up to start chatting with the AI assistant.")
    st.write("This AI agent can:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📧 Email Access")
        st.write("Read and search through your Gmail inbox")
    
    with col2:
        st.markdown("### 🌐 Web Scraping")
        st.write("Extract information from websites")
    
    with col3:
        st.markdown("### 🧠 Memory")
        st.write("Remember important information across conversations")
    
    st.markdown("""
    <div class="auth-message">
    <strong>Note:</strong> You'll need to authorize Gmail access when first using email features.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # This won't run in Streamlit, but kept for consistency
    pass