from fastapi import FastAPI
from agent import ProductionAgent
from models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse, ErrorResponse


agent = ProductionAgent()


app = FastAPI()

@app.get("/")
def home():
    return {"message": "running..."}

@app.post("/chat")
def chat(body: ChatRequest):
    result = agent.invoke(body.message)
    return result
