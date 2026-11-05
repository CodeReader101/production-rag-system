from config import get_settings
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable
from langchain_openrouter import ChatOpenRouter


# add the moniter logger

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:

    def __init__(self):

        settings = get_settings()

        self.primary_model = ChatOpenRouter(
            api_key=settings.openrouter_api_key,
            model = settings.primary_model,
            temperature=0,
            timeout=30000, # 30 seconds in milliseconds, openrouter accepts time in milliseconds
            max_retries=0
        )

        self.fallback_model = ChatOpenRouter(
            api_key=settings.openrouter_api_key,
            model=settings.fallback_model,
            temperature=0,
            timeout=30000, # 30 seconds in milliseconds
            max_retries=0,
        )
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):

        def process_message(state: AgentState) -> dict:
            try:
                response = self.primary_model.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                    "retry_count": state["retry_count"] + 1
                }

        def try_fallback(state: AgentState) -> dict:
            try:
                response = self.fallback_model.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback"
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": ""
                }

        def handle_error(state: AgentState) -> dict:
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
                    ))
                ],
                "model_used": "error_handler"
            }

        def route_after_process(state: AgentState) -> dict:
            if state["error"] is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "process"
            else: 
                return "fallback"

        def route_after_fallback(state: AgentState) -> dict:
            if state["error"] is None:
                return "done"
            else:
                return "error"

        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process", 
            route_after_process, 
            {"done": END, "process": "process", "fallback": "fallback"}
        )
        graph.add_conditional_edges(
            "fallback", 
            route_after_fallback, 
            {"done": END, "error": "error"}
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }