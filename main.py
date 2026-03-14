from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from agents import create_fascicolo_crew
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="EgoVoid AI Backend - GDS-01")

# CORS per Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://egovoid.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    sender: str
    content: str

class FascicoloRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def read_root():
    return {
        "status": "GDS-01 Backend Running",
        "model": "Groq llama-3.3-70b-versatile",
        "version": "GDS-01 v2.0 - Specchio/Eco"
    }

@app.post("/fascicolo")
async def generate_fascicolo(request: FascicoloRequest):
    """Genera fascicolo psicologico con GDS-01 multi-agent"""
    
    # Converti messaggi in trascrizione
    trascrizione = "\n\n".join([
        f"{msg.sender.upper()}: {msg.content}" 
        for msg in request.messages
    ])
    
    print(f"\n🔍 Analizzando {len(request.messages)} messaggi...")
    
    try:
        # Crea crew GDS-01 e genera
        crew = create_fascicolo_crew(trascrizione)
        result = crew.kickoff()
        
        print(f"✅ Fascicolo generato!")
        
        return {
            "fascicolo": str(result),
            "messages_analyzed": len(request.messages),
            "model": "GDS-01 v2.0"
        }
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {
            "error": str(e),
            "messages_analyzed": len(request.messages)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
