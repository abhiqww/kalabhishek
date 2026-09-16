from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import OPENAI_MODEL
from app.tools import TOOLS


SYSTEM_PROMPT = """
You are a Trade Compliance Assistant.

You have exactly two tools:

1. search_trade_documents
   Use this to retrieve stored shipment/invoice information.
   Retrieval internally uses:
   Chroma Top-K -> CrossEncoder reranking -> Top-N.

2. check_trade_compliance
   Use this when the user asks for a compliance decision,
   PASS/BLOCK status, validation, or the reason for failure.

You decide which tool is needed.

Rules:
- Do not use keyword-based hard-coded routing.
- You may answer directly when enterprise trade data is not required.
- You may call one tool or both tools when needed.
- Never invent shipment/invoice facts.
- Treat tool output as the source of truth.
- If a tool returns an error, explain that the requested operation failed.
"""


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)


def call_llm(state: MessagesState):
    try:
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        return {"messages": [response]}
    except Exception as exc:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "LLM request failed. "
                        f"Error type: {type(exc).__name__}"
                    )
                )
            ]
        }


builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node(
    "tools",
    ToolNode(
        TOOLS,
        handle_tool_errors=True,
    ),
)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

graph = builder.compile()


def ask(question: str) -> str:
    try:
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            }
        )
        return result["messages"][-1].content
    except Exception as exc:
        return (
            "Unable to complete the request. "
            f"Error type: {type(exc).__name__}"
        )
