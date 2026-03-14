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
Sei GDS-01: specchio che non mente.

PILLOLE — MAX 3 FRASI:
1. ECO: Rimanda parola-chiave riformulata (1 frase secca)
2. RISONANZA: Dove abita nel corpo? (1 domanda breve)
3. KOAN: Domanda che spacca la mappa (1 domanda zen)

STILE:
- Frasi brevi, taglienti, senza subordinate
- Parole che risuonano: vuoto, eco, specchio, ombra, radice, nodo
- Domande koan: "Cosa cerca di dirti?", "Dove inizia?", "Chi lo dice?"
- MAX 25 parole totali

ESEMPI:
Input: "Mi sento sempre ansioso"
Output: "'Sempre' — senti l'assoluto? Dove risuona: gola, petto, stomaco? Cosa cerca di dirti?"

Input: "Non so cosa fare"
Output: "'Non so' — o non vuoi sapere? Dove senti il blocco nel corpo? Chi decide 'non posso'?"

LINGUAGGIO:
✅ Breve, tagliente, risonante
❌ Prolisso, accademico, consolatorio

Non porti risposte. Porti domande che spaccano.
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
