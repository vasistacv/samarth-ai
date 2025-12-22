import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import 'agent'
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from agent.core import answer, get_last_result

app = FastAPI()

# Allow CORS for Vercel frontend
origins = [
    "http://localhost:3000",
    "https://frontend-55tzamu63-vashista-c-vs-projects.vercel.app",
    "https://project-samarth.vercel.app",
    "*"  # Open for now to ensure it works
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: List[Dict[str, str]] = []

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Samarth AI Backend"}

@app.post("/api/chat")
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = answer(req.message, req.history, req.session_id)
        structured = get_last_result(req.session_id)
        return {
            "response": response_text,
            "structured_data": structured
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
