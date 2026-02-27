import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from schema.States.chat_state import ChatState

load_dotenv()

def get_workflow(tool_node, agent_node):
    """
    A workflow which has only two nodes agent and tool node.

    If agent will need any information then it must call the tool node iterative in order to get appropriate and relevent information from tools.
    """
    # Define graph
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node('agent_node', agent_node)
    graph.add_node('tools', tool_node)

    # Draw edges
    graph.add_edge(START, 'agent_node')
    graph.add_conditional_edges('agent_node', tools_condition)
    graph.add_edge('tools', 'agent_node')

    # Compile it
    workflow = graph.compile()

    # return
    return workflow