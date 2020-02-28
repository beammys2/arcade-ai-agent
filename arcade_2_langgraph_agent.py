from langchain_arcade import ToolManager
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
import os

# Load in the environment variables
load_dotenv()
arcade_api_key = os.environ.get("ARCADE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_choice = os.environ.get("MODEL_CHOICE")
email = os.environ.get("EMAIL")

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


#### Workflow ####


# Function to invoke the model and get a response
async def call_agent(state: MessagesState, writer):
    messages = state["messages"]
    
    # Stream tokens using astream
    full_content = ""
    tool_calls = []
    
    async for chunk in model_with_tools.astream(messages):
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
def authorize(state: MessagesState, config: RunnableConfig, writer):
    user_id = config["configurable"].get("user_id")
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        if not manager.requires_auth(tool_name):
            continue
        auth_response = manager.authorize(tool_name, user_id)
        if auth_response.status != "completed":
            # Stream the authorization URL to the user
            writer(f"\n🔐 Authorization required for {tool_name}\n")
            writer(f"Visit the following URL to authorize: {auth_response.url}\n")
            writer("Waiting for authorization...\n")

            # wait for the user to complete the authorization
            # and then check the authorization status again
            manager.wait_for_auth(auth_response.id)
            if not manager.is_authorized(auth_response.id):
                # node interrupt?
                raise ValueError("Authorization failed")

    return {"messages": []}


# Builds the LangGraph workflow with memory
def build_graph():
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

    # Set up memory for checkpointing the state
    memory = MemorySaver()

    # Compile the graph with the checkpointer
    graph = workflow.compile(checkpointer=memory)
    return graph


async def main():
    graph = build_graph()
    
    # Define the input messages from the user
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": "What emails do I have in my inbox from today?",
            }
        ],
    }

    # Configuration with thread and user IDs for authorization purposes
    config = {"configurable": {"thread_id": "4", "user_id": email}}

    # Run the graph and stream the outputs
    async for chunk in graph.astream(inputs, config=config, stream_mode="values"):
        # Pretty-print the last message in the chunk
        chunk["messages"][-1].pretty_print()
        

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())