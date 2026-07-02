from fastapi import FastAPI

# from agent import RecommenderAgent
# from catalog import load_catalog
# from llm_client import LLMClient
# from retriever import CatalogRetriever
# from schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent import RecommenderAgent
from app.catalog import load_catalog
from app.llm_client import LLMClient
from app.retriever import CatalogRetriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(title="SHL Assessment Recommender")

# Loaded once at startup, kept in memory -- 377 items, no DB needed.
_catalog = load_catalog()
_retriever = CatalogRetriever(_catalog, use_semantic=True)
_llm = LLMClient()  # reads ANTHROPIC_API_KEY from env
_agent = RecommenderAgent(_catalog, _retriever, _llm)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = _agent.handle_turn(messages)
    return ChatResponse(**result)

@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommendation API",
        "docs": "/docs",
        "health": "/health"
    }
