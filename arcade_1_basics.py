from langchain_arcade import ToolManager
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
import os

# Load in the environment variables
load_dotenv()
arcade_api_key = os.environ.get("ARCADE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_choice = os.environ.get("MODEL_CHOICE")
email = os.environ.get("EMAIL")

# 2) Create an ToolManager and fetch/add tools/toolkits
manager = ToolManager(api_key=arcade_api_key)
 
# Get all tools from the "Gmail" and "Asana" toolkit
tools = manager.get_tools(toolkits=["Gmail", "Asana"])
print(manager.tools)

# 3) Get StructuredTool objects for langchain
lc_tools = manager.to_langchain()

# 4) Create a ChatOpenAI model and bind the Arcade tools.
model = ChatOpenAI(model=model_choice, api_key=openai_api_key)
bound_model = model.bind_tools(lc_tools)

# 5) Use MemorySaver for checkpointing.
memory = MemorySaver()

# 5) Create a ReAct-style agent from the prebuilt function.
graph = create_react_agent(model=bound_model, tools=lc_tools, checkpointer=memory)

# 6) Provide basic config and a user query.
# Note: user_id is required for the tool to be authorized
config = {"configurable": {"thread_id": "1", "user_id": email}}
user_input = {"messages": [("user", "Grab all emails in my inbox. Then tell me what Cole said about the AI SaaS ideas")]}

# 7) Stream the agent's output. If the tool is unauthorized, it may trigger interrupts
for chunk in graph.stream(user_input, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# if we were interrupted, we can check for interrupts in state
current_state = graph.get_state(config)
if current_state.tasks:
    for task in current_state.tasks:
        if hasattr(task, "interrupts"):
            for interrupt in task.interrupts:
                print(interrupt.value)