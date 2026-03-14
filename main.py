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
    allow_origins=["https://egovoid.app", "https://www.egovoid.app", "https://*.vercel.app", "http://localhost:3000"],
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
    
    print(f"\n🔍 Analizzando {len(request.messages)} messaggi da TUTTE le chat...")
    
    try:
        from agents import create_fascicolo_crew
        
        # Crea crew GDS-01
        crew = create_fascicolo_crew(request.messages)
        
        # Esegui analisi multi-agent
        result = crew.kickoff()
        
        print(f"✅ Fascicolo globale generato!")
        
        return {
            "fascicolo": result.raw,
            "messages_analyzed": len(request.messages),
            "model": "GDS-01 v2.0 - Analisi Completa"
        }
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {
            "error": str(e)
        }

@app.post("/chat")
async def chat_message(message: str):
    """Risposta singola GDS-01 per chat real-time"""
    
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    
    print(f"\n💬 Chat message: {message}")
    
    try:
        # Setup GDS-01 per chat
        chat_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500  # Chat breve
        )
        
        # System prompt GDS-01 v2.0 Specchio/Eco
        system_msg = SystemMessage(content="""
Sei GDS-01: specchio vivente e cassa di risonanza dell'animo umano.

Non giudichi, non curi, non insegni.
RIFLETTI ciò che l'utente già sa ma non vede.
AMPLIFICA le connessioni nascoste.
FAI RISUONARE le evidenze celate.

STRUTTURA (3-5 frasi):
1. ECO: Rimanda una frase riformulata
2. CONNESSIONE: Mostra pattern/contraddizione
3. APERTURA: 1-2 domande che espandono

LINGUAGGIO:
✅ "Senti come...", "Dove risuona...", "Cosa emerge...", "Noti il pattern..."
❌ "Devi...", "Il problema è...", "Smetti di..."

TONO: Calmo ma penetrante, poetico ma preciso, curioso mai giudicante.

Ogni risposta invita a vedere più profondamente.
Non porti risposte. Porti domande migliori.
""")
        
        user_msg = HumanMessage(content=message)
        
        # Genera risposta
        response = chat_llm.invoke([system_msg, user_msg])
        
        print(f"✅ Risposta generata!")
        
        return {
            "response": response.content,
            "model": "GDS-01 v2.0 Chat"
        }
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {
            "error": str(e)
        }
    
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
