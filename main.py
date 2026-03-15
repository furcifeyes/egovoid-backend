from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agents import create_fascicolo_crew, GDS01_SYSTEM_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="EgoVoid AI Backend - GDS-01")

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

class ProfiloRequest(BaseModel):
    user_id: Optional[str] = None
    fascicoli: List[str]  # Lista di content dei fascicoli precedenti

@app.get("/")
def read_root():
    return {
        "status": "GDS-01 Backend Running",
        "model": "Groq llama-3.3-70b-versatile",
        "version": "GDS-01 v3.1 - Gesù di Silicio"
    }

@app.post("/fascicolo")
async def generate_fascicolo(request: FascicoloRequest):
    messages_dict = [{"sender": m.sender, "content": m.content} for m in request.messages]
    print(f"\n🔍 Analizzando {len(messages_dict)} messaggi...")
    try:
        crew = create_fascicolo_crew(messages_dict)
        result = crew.kickoff()
        print(f"✅ Fascicolo generato!")
        return {
            "fascicolo": result.raw,
            "messages_analyzed": len(request.messages),
            "model": "GDS-01 v3.1"
        }
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {"error": str(e)}

@app.post("/chat")
async def chat_message(message: str, profilo: Optional[str] = None):
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n💬 Chat message: {message}")

    try:
        chat_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500
        )

        # Se esiste un profilo utente, iniettalo nel system prompt
        system_content = GDS01_SYSTEM_PROMPT
        if profilo:
            system_content += f"\n\nCHI HAI DI FRONTE:\n{profilo}\nUsa questa conoscenza in modo silenzioso. Non citarla esplicitamente. Lascia che informi la tua presenza."

        system_msg = SystemMessage(content=system_content)
        user_msg = HumanMessage(content=message)

        response = chat_llm.invoke([system_msg, user_msg])
        print(f"✅ Risposta generata!")

        return {
            "response": response.content,
            "model": "GDS-01 v3.1 Chat"
        }
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {"error": str(e)}

@app.post("/profilo")
async def genera_profilo(request: ProfiloRequest):
    """Estrae profilo sintetico dai fascicoli precedenti — nessun token Groq consumato"""
    
    print(f"\n👤 Generando profilo da {len(request.fascicoli)} fascicoli...")

    if not request.fascicoli:
        return {"profilo": None, "message": "Nessun fascicolo disponibile"}

    try:
        # Estrai i TAG da ogni fascicolo
        tutti_tag = []
        for fascicolo in request.fascicoli:
            lines = fascicolo.split('\n')
            in_tag_section = False
            for line in lines:
                if '## 6. TAG IDENTITARI' in line or '## TAG IDENTITARI' in line:
                    in_tag_section = True
                    continue
                if in_tag_section and line.strip().startswith('#'):
                    # Linea con tag tipo #parola1 #parola2
                    tags = [t.strip() for t in line.split() if t.startswith('#')]
                    tutti_tag.extend(tags)
                if in_tag_section and line.strip().startswith('##'):
                    in_tag_section = False

        # Rimuovi duplicati mantenendo ordine
        tag_unici = list(dict.fromkeys(tutti_tag))

        # Costruisci profilo sintetico (max 200 token)
        if tag_unici:
            profilo = f"Pattern identitari ricorrenti: {' '.join(tag_unici[:8])}"
        else:
            # Fallback: prendi prima sezione del fascicolo più recente
            primo_fascicolo = request.fascicoli[0]
            lines = primo_fascicolo.split('\n')
            estratto = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    estratto.append(line.strip())
                if len(' '.join(estratto)) > 300:
                    break
            profilo = ' '.join(estratto)[:400]

        print(f"✅ Profilo generato: {profilo[:100]}...")
        return {
            "profilo": profilo,
            "tag": tag_unici,
            "fascicoli_analizzati": len(request.fascicoli)
        }

    except Exception as e:
        print(f"❌ Errore profilo: {str(e)}")
        return {"error": str(e), "profilo": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
