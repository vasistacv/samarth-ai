from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import uvicorn
import os
from agent.core import answer, get_last_result

app = FastAPI(title="Samarth AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    structured_data: Optional[Dict[str, Any]] = None

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Samarth AI Backend"}

@app.post("/chat", response_model=ChatResponse)
def search(request: ChatRequest):
    try:
        # Get the textual answer
        response_text = answer(request.message, request.session_id)
        
        # Check if there is structured data associated with this session/response
        # The answer() function returns a string marker "[[STRUCTURED_RESULT::<session_id>]]" 
        # if it generated a table.
        structured_data = None
        if "[[STRUCTURED_RESULT::" in response_text:
            # Clean up the marker from the text if you want, or keep it.
            # Usually the frontend will hide the marker and show the table.
            # Here we fetch the structured data to send along directly.
            structured_data = get_last_result(request.session_id)
            
        return ChatResponse(
            response=response_text,
            structured_data=structured_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
