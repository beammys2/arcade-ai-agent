"""
Beautiful streaming CLI for Arcade AI Agent using LangGraph.
"""

import asyncio
import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage

from arcade_2_langgraph_agent import build_graph
from dotenv import load_dotenv

load_dotenv()
console = Console()


async def stream_agent_response(user_input: str, graph, config: dict):
    """Stream agent response with proper handling of authorization URLs."""
    
    # Prepare input
    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    response_buffer = ""
    auth_shown = False
    
    # Stream the response
    console.print("\n[bold blue]Assistant:[/bold blue] ", end="")
    
    async for chunk in graph.astream(inputs, config=config, stream_mode="custom"):
        if isinstance(chunk, str):
            # Check if this is an authorization message
            if "🔐 Authorization required" in chunk and not auth_shown:
                # Extract and display authorization nicely
                auth_shown = True
                console.print()  # New line
                
                # Extract URL if present
                url_match = re.search(r'(https?://[^\s]+)', chunk)
                if url_match:
                    auth_url = url_match.group(1)
                    console.print("\n" + "="*60)
                    console.print(Panel(
                        f"[bold yellow]🔐 Authorization Required[/bold yellow]\n\n"
                        f"Please visit this URL to authorize:\n"
                        f"[bold cyan][link]{auth_url}[/link][/bold cyan]\n\n"
                        f"[dim]The agent will wait for you to complete authorization...[/dim]",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
                    console.print("="*60 + "\n")
                else:
                    console.print(chunk, end="")
            else:
                console.print(chunk, end="")
                response_buffer += chunk
        elif isinstance(chunk, bytes):
            decoded = chunk.decode('utf-8')
            console.print(decoded, end="")
            response_buffer += decoded
    
    return response_buffer


async def main():
    """Main conversation loop."""
    
    # Welcome
    console.print(Panel(
        "[bold blue]🎮 Arcade AI Agent CLI[/bold blue]\n\n"
        "[green]Powered by LangGraph and Arcade AI[/green]\n"
        "[dim]Type 'exit' to quit, 'help' for commands[/dim]",
        style="blue",
        padding=(1, 2)
    ))
    console.print()
    
    # Build the graph
    try:
        console.print("[dim]Building agent graph...[/dim]")
        graph = build_graph()
        console.print("[green]✓ Agent ready![/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to build agent: {e}[/red]")
        return
    
    # Configuration
    email = os.environ.get("EMAIL")
    config = {
        "configurable": {
            "thread_id": "cli-session",
            "user_id": email
        }
    }
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
            
            # Handle exit
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
                
            if not user_input:
                continue
            
            # Handle help
            if user_input.lower() == 'help':
                help_text = Panel(
                    "[bold]Available Commands:[/bold]\n\n"
                    "• Ask about emails: 'What emails do I have from today?'\n"
                    "• Search emails: 'Find emails about project updates'\n" 
                    "• Scrape websites: 'Get information from example.com'\n"
                    "• General queries: Ask anything!\n\n"
                    "[dim]The agent will request authorization when needed[/dim]\n"
                    "[dim]Type 'exit' to quit[/dim]",
                    title="Help",
                    style="blue"
                )
                console.print(help_text)
                continue
            
            # Add to history
            conversation_history.append(f"User: {user_input}")
            
            # Stream the response
            response = await stream_agent_response(user_input, graph, config)
            
            # Add response to history
            if response:
                conversation_history.append(f"Assistant: {response}")
            
            # Add some spacing
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            continue
            
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {e}[/red]")