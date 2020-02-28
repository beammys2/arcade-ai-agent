# Arcade AI Agent with LangGraph

A comprehensive tutorial project demonstrating how to build production-ready AI agents using [Arcade AI](https://arcade.dev) for tool management and [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration and long term memory. This repository progressively builds from basic tool usage to a full-featured agent with memory, authentication, and a Streamlit interface.

## 🚀 Overview

This project showcases the integration of Arcade AI's powerful tool ecosystem with LangGraph's advanced agent framework. The agent has access to:

- **Gmail Integration**: Read, search, and manage emails
- **Asana Integration**: Access and manage tasks, projects, and team collaboration
- **Long-term Memory**: Persistent conversation memory using PostgreSQL
- **Production Features**: Authentication, session management, and robust error handling

The codebase follows a progressive learning approach, evolving from simple tool usage to enterprise-ready deployment patterns.

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL database (for memory and checkpointing)
- OpenAI API key
- Arcade API key
- Gmail Account
- Asana Account

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/coleam00/arcade-ai-agent.git
cd arcade-ai-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Get your Arcade API key: https://docs.arcade.dev/home/api-keys
ARCADE_API_KEY=your_arcade_api_key

# OpenAI API key from https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key

# Model selection
MODEL_CHOICE=gpt-4.1-mini

# Your email for tool authorization
EMAIL=your.email@example.com

# PostgreSQL connection string
DATABASE_URL=postgresql://user:password@host:port/database

# Supabase configuration (for Streamlit auth)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

## 🎯 Tutorial Progression

This repository is structured as a step-by-step tutorial, with each file building upon the previous:

### 1. `arcade_1_basics.py` - Foundation
**Purpose**: Introduction to Arcade AI tool integration with LangGraph

**What You'll Learn**:
- Basic tool manager setup with Gmail and Asana access
- Simple ReAct agent creation using LangGraph prebuilt functions
- Tool authorization flow and basic conversation handling

**Run**: `python arcade_1_basics.py`

### 2. `arcade_2_langgraph_agent.py` + CLI Interface - Custom Workflow
**Purpose**: Building custom LangGraph workflows with beautiful CLI interface

**What You'll Learn**:
- Custom graph implementation with agent-tools-authorization flow
- Real-time streaming token responses with Rich formatting
- Enhanced authorization handling with visual panels
- Interactive CLI conversation loop with help system
- Async/await patterns for better performance

**Run**: `python arcade_2_langgraph_cli.py`

### 3. `arcade_3_agent_with_memory.py` + Streamlit Interface - Production Memory
**Purpose**: Adding persistent memory with PostgreSQL backend and web interface

**What You'll Learn**:
- PostgreSQL checkpointer and store for persistent conversation history
- Semantic memory search and retrieval with user-specific namespaces
- Memory-aware conversation context integration
- Supabase authentication with session management
- Production web deployment with real-time streaming in Streamlit
- Responsive web UI with proper error handling

**Run**: `streamlit run arcade_3_streamlit_app.py`

## 🔧 Usage Examples

### Email Management
```python
# Ask about emails
"What emails do I have in my inbox from today?"

# Search specific content
"Find emails about project updates"

# Remember information
"Remember that the project deadline is next Friday"
```

### Asana Task Management
```python
# View tasks
"What tasks do I have assigned to me?"

# Create tasks
"Create a new task called 'Review marketing proposal'"

# Search projects
"Show me all projects I'm working on"

# Update task status
"Mark the design review task as complete"
```

### Memory Features
```python
# Store information
"Remember that John prefers meetings on Tuesdays"

# Recall information
"What do you remember about John's preferences?"

# Contextual memory
"What did we discuss about the project last week?"
```

## 🏗️ Architecture

### Core Components

#### 1. Tool Management Layer (Arcade AI)
- **ToolManager**: Centralizes tool discovery and authorization
- **Supported Tools**: Gmail, Asana, and extensible toolkit system
- **Authorization Flow**: OAuth2-based tool authentication
- **LangChain Integration**: Seamless conversion to LangChain tools

#### 2. Agent Orchestration Layer (LangGraph)
- **StateGraph**: Custom workflow definition with conditional routing
- **Nodes**: Agent reasoning, tool execution, authorization handling
- **Edges**: Control flow between different agent states
- **Streaming**: Real-time response generation and display

#### 3. Memory & Persistence Layer
- **PostgreSQL Backend**: Production-grade data persistence
- **Checkpointer**: Conversation state management across sessions
- **Memory Store**: Semantic search and retrieval of past interactions
- **User Isolation**: Namespace-based memory separation

### Data Flow Architecture

```
User Input → Interface Layer → LangGraph Agent → Tool Authorization → Tool Execution → Memory Storage → Response Streaming → User Interface
```

#### Detailed Flow:

1. **Input Processing**: User query received through CLI/Web interface
2. **Context Loading**: Relevant memories retrieved from PostgreSQL store
3. **Agent Reasoning**: LLM processes input with context and available tools
4. **Tool Selection**: Agent decides which tools to use based on query
5. **Authorization Check**: Verify user permissions for selected tools
6. **Tool Execution**: Execute authorized tools with user credentials
7. **Memory Update**: Store interaction and results in persistent memory
8. **Response Generation**: Stream formatted response back to user
9. **Session Persistence**: Save conversation state for future sessions

## 🚀 Deployment

### Development
```bash
# Basic implementation
python arcade_1_basics.py

# Custom workflow
python arcade_2_langgraph_agent.py

# CLI interface
python arcade_2_langgraph_cli.py

# Memory-enabled agent
python arcade_3_agent_with_memory.py

# Web interface
streamlit run arcade_3_streamlit_app.py
```

## 📚 Key Learning Outcomes

By working through this tutorial, you'll learn:

1. **Tool Integration Patterns**: How to integrate external APIs using Arcade AI
2. **Agent Architecture**: Building custom workflows with LangGraph
3. **Memory Management**: Implementing persistent, searchable memory systems
4. **Authorization Flows**: Handling OAuth2 and user permissions
5. **Interface Development**: Creating both CLI and web interfaces
6. **Production Deployment**: Scaling from prototype to production
7. **Async Programming**: Managing concurrent operations and streaming

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- [Arcade AI Documentation](https://docs.arcade.dev/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Supabase Documentation](https://supabase.com/docs)