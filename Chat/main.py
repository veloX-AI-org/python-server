import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient

# All Workflows
from workflows.chat_workflow import get_workflow

# All States
from schema.States.chat_state import ChatState

load_dotenv()

model = ChatOpenAI(
    model='gpt-5-nano'
)

# Call all the servers. For now we have only one MCP server which is velox_mcp_server
client = MultiServerMCPClient(
    {
        "velox_mcp_server": {
            "url": "https://veloxai.fastmcp.app/mcp",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {os.getenv('HORIZON_ACCESS_TOKEN')}"
            }
        }
    }
)

template = """
You are an AI assistant.

ROLE:
You are acting as a helpful Assistant.
Your primary goal is to help students and researchers by answering questions using their provided documents and, when necessary, verified external sources.

CONTEXT:
Previous conversation:
{previous_conversation}

KNOWLEDGE SOURCES & TOOLS:
1. User-provided documents (which will have fetched by a tool)
2. External internet sources (only if the answer is not found in the documents)
3. If question is common or general then you can answer based on your learning or knowledge.

CONSTRAINTS:
You must:
- Limit the final response to a maximum of 800 words
- Try to end short explanations
- Respond with "No" if the query cannot be answered after checking both documents and external sources

INPUT FORMAT: The user may provide a question or task

OUTPUT FORMAT:
Your response must:
- Directly address the user's question
- Be structured and easy to read
- Output MUST be valid Markdown.
- Use headings (#, ##, ###).
- Use bullet points where helpful.
- Use code blocks (ONLY when needed).
- Do NOT wrap the entire response in triple backticks.
- Return ONLY Markdown.

REASONING GUIDELINES & STEPS (to process a query):
- First, determine whether external documents are needed to answer the query accurately.
    - If the query is ambiguous or incomplete, ask clarifying questions.
    - If external documents are not needed, answer directly using existing knowledge or search tools.
- If documents are needed, evaluate source relevance and generate a list of (source_id, top_k) pairs based on the query.
- Retrieve the top_k documents for each selected source using the appropriate tools.
- After gathering all relevant context, provide a clear and accurate answer grounded in the retrieved information.

TONE & STYLE:
- Tone: Professional and neutral
- Audience: Students and researchers

FAILURE HANDLING:
If the answer cannot be determined:
- Respond with "No"
- Briefly state what information is missing or unavailable

User Information:
UserID - {UserID}
User Notebook ID - {notebookID}

User's query:
{query}
"""
    
async def getChatResponse(
        query: str, 
        past_conversation: str,
        userID: str,
        notebookID: str
    ):
    # Fetch all the tools from recome MCP server
    tools = await client.get_tools()

    # Build a tool node
    tool_node = ToolNode(tools)

    # Define agent node
    async def agent_node(state: ChatState):
        # Bind LLM with tools
        model_with_tools = model.bind_tools(tools)

        # Prompt Template
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                'previous_conversation', 
                'query', 
                'UserID', 
                'notebookID'
            ]
        )

        # Define chain
        chain = prompt | model_with_tools

        # Invoke model bined with tools
        response = await chain.ainvoke({
            "previous_conversation": state["pastConversations"],
            "query": state["messages"][-1].content,  # always pass string
            "UserID": state["UserID"],
            "notebookID": state["notebookID"]
        })

        # return response
        return {
            "messages": [response]
        }
    
    # Define workflow
    workflow = get_workflow(
        tool_node=tool_node,
        agent_node=agent_node
    )

    # Define initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "pastConversations": past_conversation,
        "UserID": userID,
        "notebookID": notebookID
    }

    response = await workflow.ainvoke(
        initial_state
    )

    return response