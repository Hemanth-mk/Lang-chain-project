# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated, Literal
from operator import add
import os
from dotenv import load_dotenv

# LangGraph & LangChain cores
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field
import time  # 🚀 ADD THIS IMPORT AT THE TOP OF YOUR FILE
import uuid
load_dotenv()

# ── 1. DEFINE THE SHARED GRAPH STATE MEMORY ──────────────────────────
class ResearchState(TypedDict):
    query: str
    plan: str
    search_results: Annotated[list[str], add]   # Automatically accumulates data
    summary: str
    code_snippets: Annotated[list[str], add]     # Automatically accumulates data
    final_report: str
    next_agent: str
    iteration: int
    errors: Annotated[list[str], add]

# Initialize high-speed, zero-RAM cloud models via Groq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# ── 2. AGENT LOGIC OBJECTS ───────────────────────────────────────────
class SupervisorDecision(BaseModel):
    next_agent: str = Field(description="Must be one of: research, summarize, coding, report, end")
    reasoning: str = Field(description="Brief explanation why this step was chosen")

from langchain_core.output_parsers import SimpleJsonOutputParser # 🚀 More resilient JSON parser

class SupervisorAgent:
    def __init__(self):
        # We explicitly use ChatGroq (or ChatOpenAI)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an elite supervisor orchestrating a multi-agent research pipeline.
Your job is to read the current progress markers and intelligently route execution to the best next worker agent.

Specialized Agents Available:
- research: Gathers facts and search queries based on the user topic.
- summarize: Condenses raw data inputs and research files into structured takeaways.
- coding: Writes working, error-free programming script snippets.
- report: Compiles all previous findings, summaries, and code into a single professional markdown paper.
- end: The final report is complete and polished. Stop the pipeline.

Guidelines:
1. Always establish a research foundation first.
2. Only run 'coding' if the query explicitly asks for code or technical building steps.
3. Once all components are satisfied, always route to 'report' before routing to 'end'.

⚠️ CRITICAL SAFETY RULE: You must output ONLY valid raw JSON text code. 
Do NOT include any introduction text, markdown code-blocks (like ```json), or extra sentences before or after the JSON structure.
Your response MUST start with {{ and end with }}.

Return JSON strictly matching this structure:
{{"next_agent": "research", "reasoning": "Explain your choice here"}}"""),
            ("human", """User Query: {query}
Has Search Results: {has_results}
Has Summary Data: {has_summary}
Has Code Snippets: {has_code}
Has Final Report: {has_report}
Current Iteration: {iteration}""")
        ])
        
        # 🚀 FIX: SimpleJsonOutputParser strips text outside brackets much better than JsonOutputParser
        self.chain = self.prompt | llm | SimpleJsonOutputParser()

    def decide(self, state: ResearchState) -> dict:
        result = self.chain.invoke({
            "query": state["query"],
            "has_results": len(state.get("search_results", [])) > 0,
            "has_summary": bool(state.get("summary")),
            "has_code": len(state.get("code_snippets", [])) > 0,
            "has_report": bool(state.get("final_report")),
            "iteration": state["iteration"]
        })
        
        # Fallback guardrail: If the parser returned a raw string by accident, keep it safe
        if isinstance(result, str):
            import json
            # Extract just the JSON part manually if needed
            start = result.find("{")
            end = result.rfind("}") + 1
            result = json.loads(result[start:end])
            
        return result

# ── 3. STATE GRAPH NODES (WORKER LOOPS) ──────────────────────────────

# ── 3. UPDATED STATE GRAPH NODES (WITH INTEGRATED RATE MITIGATION) ──

def supervisor_node(state: ResearchState) -> ResearchState:
    # Introduce a tiny 1.5-second cooling delay so we don't spam Groq back-to-back
    time.sleep(1.5)
    decision = SupervisorAgent().decide(state)
    return {**state, "next_agent": decision["next_agent"], "iteration": state["iteration"] + 1}

def route_to_agent(state: ResearchState) -> Literal["research", "summarize", "coding", "report", "end"]:
    if state["iteration"] > 8:
        return "end"
    return state["next_agent"]

def research_node(state: ResearchState) -> ResearchState:
    time.sleep(1.5)  # 🕒 Cool down API clock counter
    prompt = f"Act as the Research Agent. Brainstorm 3 distinct brief search vectors or data points to analyze: '{state['query']}'."
    res = llm.invoke(prompt)
    return {**state, "search_results": [res.content]}

def summarize_node(state: ResearchState) -> ResearchState:
    time.sleep(1.5)  # 🕒 Cool down API clock counter
    
    # FIX: Slice and keep ONLY the last 3 research findings to stop context bloat (>6000 tokens)
    limited_context = state["search_results"][-3:]
    context_text = "\n".join(limited_context)
    
    prompt = f"Act as the Summarize Agent. Condense these raw research findings into 3 high-impact analytical bullets:\n\n{context_text}"
    res = llm.invoke(prompt)
    return {**state, "summary": res.content}

def coding_node(state: ResearchState) -> ResearchState:
    time.sleep(1.5)  # 🕒 Cool down API clock counter
    prompt = f"Act as the Coding Agent. Write a completely working, clean Python code block example matching the request: '{state['query']}'."
    res = llm.invoke(prompt)
    return {**state, "code_snippets": [res.content]}

def report_node(state: ResearchState) -> ResearchState:
    time.sleep(1.5)  # 🕒 Cool down API clock counter
    
    # FIX: Guard snippet state length securely
    limited_snippets = state.get("code_snippets", [])[-2:]
    snippets_text = "\n".join(limited_snippets)
    
    prompt = f"""Act as the Report Agent. Compile a long-form, comprehensive, beautifully formatted technical report based on:
Query: {state['query']}
Summary: {state['summary']}
Code Block: {snippets_text}
Format beautifully using clean markdown with bold headers and bullet structures."""
    res = llm.invoke(prompt)
    return {**state, "final_report": res.content}

# ... keep the rest of your build_graph() and FastAPI endpoints exactly the same ...

# ── 4. BUILD AND COMPILE THE LANGGRAPH STATE ENGINE ──────────────────
def build_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)
    
    # Register our functional agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("report", report_node)
    
    workflow.set_entry_point("supervisor")
    
    # Wire conditional branching path rules
    workflow.add_conditional_edges("supervisor", route_to_agent, {
        "research": "research",
        "summarize": "summarize",
        "coding": "coding",
        "report": "report",
        "end": END
    })
    
    # Force every individual worker node to loop directly back to supervisor
    for node in ["research", "summarize", "coding", "report"]:
        workflow.add_edge(node, "supervisor")
        
    # FIX: Native local memory saver handles state checkpoints instantly without RAM overhead
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

compiled_graph = build_graph()

# ── 5. FASTAPI NETWORK LAYER ─────────────────────────────────────────
app = FastAPI(title="Autonomous LangGraph Orchestration Core")

class UserQueryRequest(BaseModel):
    query: str

@app.post("/run_research")
def run_research_pipeline(request: UserQueryRequest):
    # Initialize clean, base graph state values
    initial_state = {
        "query": request.query,
        "plan": "",
        "search_results": [],
        "summary": "",
        "code_snippets": [],
        "final_report": "",
        "next_agent": "",
        "iteration": 0,
        "errors": []
    }
    
    # Execute graph state machine using a unique thread tracking configuration
    config = {"configurable": {"thread_id": str(uuid.uuid4() if 'uuid' in globals() else 101)}}
    final_output = compiled_graph.invoke(initial_state, config=config)
    
    return {
        "report": final_output.get("final_report", "Pipeline terminated without report."),
        "iterations": final_output.get("iteration", 0),
        "summary": final_output.get("summary", "No summary captured.")
    }