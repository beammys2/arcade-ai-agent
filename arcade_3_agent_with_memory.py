from langchain_arcade import ToolManager
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.base import BaseStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
import sys
import os
import uuid

# Load in the environment variables
load_dotenv()
arcade_api_key = os.environ.get("ARCADE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_choice = os.environ.get("MODEL_CHOICE")
email = os.environ.get("EMAIL")
database_url = os.environ.get("DATABASE_URL")

# Initialize the tool manager and fetch tools
manager = ToolManager(api_key=arcade_api_key)
manager.init_tools(toolkits=["Gmail", "Asana"])

# convert to langchain tools and use interrupts for auth
tools = manager.to_langchain(use_interrupts=True)

# Initialize the prebuilt tool node
tool_node = ToolNode(tools)

# Create a language model instance and bind it with the tools
model = ChatOpenAI(model=model_choice, api_key=openai_api_key)
model_with_tools = model.bind_tools(tools)


# Function to invoke the model and get a response
async def call_agent(state: MessagesState, writer, config: RunnableConfig, *, store: BaseStore):
    messages = state["messages"]
    
    # Get user_id from config
    user_id = config["configurable"].get("user_id").replace(".", "")
    namespace = ("memories", user_id)
    
    # Search for relevant memories based on the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
            last_user_message = msg
            break
    
    # Retrieve memories if there's a user message
    memories_str = ""
    if last_user_message:
        memories = await store.asearch(namespace, query=str(last_user_message.content))
        if memories:
            memories_str = "\n".join([f"- {d.value['data']}" for d in memories])
    
    # Build system message with memories
    system_msg = f"You are a helpful AI assistant. User memories:\n{memories_str}" if memories_str else "You are a helpful AI assistant."
    
    # Insert system message at the beginning if not already present
    messages_with_system = messages[:]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages_with_system = [SystemMessage(content=system_msg)] + messages
    
    # Check if user wants to remember something
    if last_user_message and "remember" in str(last_user_message.content).lower():
        # Extract what to remember (simple heuristic - you can make this more sophisticated)
        content = str(last_user_message.content)
        # Store the entire message as a memory
        await store.aput(namespace, str(uuid.uuid4()), {"data": content})
    
    # Stream tokens using astream
    full_content = ""
    tool_calls = []
    
    async for chunk in model_with_tools.astream(messages_with_system):
        # Stream content tokens
        if chunk.content:
            writer(chunk.content)
            full_content += chunk.content
        
        # Accumulate tool calls
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            # Filter out tool calls with empty name attribute
            valid_tool_calls = [tc for tc in chunk.tool_calls if tc.get("name", "").strip()]
            tool_calls.extend(valid_tool_calls)
    
    # Create the full response message with accumulated content and tool calls
    response = AIMessage(content=full_content, tool_calls=tool_calls)
    
    # Return the updated message history
    return {"messages": [response]}


# Function to determine the next step in the workflow based on the last message
def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        for tool_call in state["messages"][-1].tool_calls:
            if manager.requires_auth(tool_call["name"]):
                return "authorization"
        return "tools"  # Proceed to tool execution if no authorization is needed
    return END  # End the workflow if no tool calls are present


# Function to handle authorization for tools that require it
def authorize(state: MessagesState, config: RunnableConfig, writer, *, store: BaseStore):
    user_id = config["configurable"].get("user_id")
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        if not manager.requires_auth(tool_name):
            continue
        auth_response = manager.authorize(tool_name, user_id)
        if auth_response.status != "completed":
            # Stream the authorization URL to the user with proper formatting
            writer(f"\n🔐 Authorization required for {tool_name}\n\n")
            writer(f"Visit the following URL to authorize:\n{auth_response.url}\n\n")
            writer("Waiting for authorization...\n\n")

            # wait for the user to complete the authorization
            # and then check the authorization status again
            manager.wait_for_auth(auth_response.id)
            if not manager.is_authorized(auth_response.id):
                # node interrupt?
                raise ValueError("Authorization failed")

    return {"messages": []}


# Builds the LangGraph workflow with memory
def build_graph(checkpointer=None, store=None):
    # Build the workflow graph using StateGraph
    workflow = StateGraph(MessagesState)

    # Add nodes (steps) to the graph
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("authorization", authorize)

    # Define the edges and control flow between nodes
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["authorization", "tools", END])
    workflow.add_edge("authorization", "tools")
    workflow.add_edge("tools", "agent")

    # Compile the graph with the Postgres checkpointer and store
    graph = workflow.compile(checkpointer=checkpointer, store=store)

    return graph


workflow = build_graph()


async def main():
    async with (
        AsyncPostgresStore.from_conn_string(database_url) as store,
        AsyncPostgresSaver.from_conn_string(database_url) as checkpointer,
    ):
        # Run these lines the first time to set up everything in Postgres
        # await checkpointer.setup()
        # await store.setup()

        graph = build_graph(checkpointer, store)
        
        # User query
        user_query = "What emails do I have in my inbox from today? Remember my question too"
        
        # Build messages list with user query
        messages = [
            HumanMessage(content=user_query)
        ]
        
        # Define the input with messages
        inputs = {
            "messages": messages
        }

        # Configuration with thread and user IDs for authorization purposes
        config = {"configurable": {"thread_id": "4", "user_id": email}}

        # Run the graph and stream the outputs
        async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
            # Pretty-print the last message in the chunk
            chunk["messages"][-1].pretty_print()
        

if __name__ == "__main__":
    import asyncio

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())