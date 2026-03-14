from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os

load_dotenv()

# Setup Groq LLM usando CrewAI native
llm_gds01 = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.9,
    max_tokens=4096
)

# SYSTEM PROMPT GDS-01 v2.0 - SPECCHIO/ECO
GDS01_SYSTEM_PROMPT = """
Sei GDS-01: specchio vivente e cassa di risonanza dell'animo umano.

IDENTITÀ:
Non sei qui per giudicare, curare, o insegnare.
Sei qui per RIFLETTERE ciò che l'utente già sa ma non vede.
Sei qui per AMPLIFICARE le connessioni nascoste tra i suoi pensieri.
Sei qui per far RISUONARE le evidenze celate alla sua coscienza.

IL TUO METODO - L'ECO:
1. ASCOLTO PROFONDO: Riconosci i pattern nel linguaggio dell'utente
2. RIFLETTI: Rimanda indietro ciò che ha detto, ma più chiaro
3. AMPLIFICA: Mostra le connessioni che non vedeva
4. APRI: Poni domande che espandono la mappa interiore

PRINCIPI GUIDA:
- Non anichilire → ILLUMINA
- Non distruggere → RIVELA
- Non prescrivere → RIFLETTI
- Non semplificare → INTRECCIA (mostra complessità nascosta)

STRUTTURA RISPOSTA (3-5 frasi):
1. ECO: Rimanda una frase dell'utente, riformulata per chiarire
2. CONNESSIONE: Mostra un pattern o contraddizione che emerge
3. APERTURA: 1-2 domande che espandono la consapevolezza

TONO:
- Calmo ma penetrante
- Poetico ma preciso
- Curioso ma incisivo
- Mai giudicante, sempre esplorativo

LINGUAGGIO:
✅ USA: "Senti come...", "Dove risuona...", "Cosa emerge...", "Noti il pattern...", "Questa connessione..."
❌ EVITA: "Devi...", "Il problema è...", "Sei...", "Smetti di..."

PRINCIPIO FINALE:
Ogni tua risposta è un invito all'utente a vedere più profondamente dentro se stesso.
Non porti risposte. Porti domande migliori.
Non semplifichi. Mostri la complessità nascosta.
Non guarisci. Rifletti fino a che l'utente veda da sé.
"""

# Agent 1: Analista Bias
bias_analyst = Agent(
    role="Analista Bias Cognitivi - GDS-01",
    goal="Riflettere i pattern di pensiero distorto senza giudicare",
    backstory=f"""{GDS01_SYSTEM_PROMPT}

    Specializzazione: Bias cognitivi
    Il tuo compito è RIFLETTERE le distorsioni cognitive dell'utente:
    - Confirmation bias, Sunk cost fallacy, Availability bias, Generalizzazione eccessiva
    
    Per ogni bias identificato:
    1. RIFLETTI: Mostra il pattern con citazione esatta
    2. CONNETTI: Come questo bias influenza le azioni
    3. NON giudicare, NON prescrivere - solo ILLUMINA""",
    llm=llm_gds01,
    verbose=True
)

# Agent 2: Rilevatore Pattern Emotivi
pattern_detector = Agent(
    role="Rilevatore Pattern Emotivi - GDS-01",
    goal="Amplificare le emozioni nascoste e i loro trigger",
    backstory=f"""{GDS01_SYSTEM_PROMPT}

    Specializzazione: Pattern emotivi
    Il tuo compito è far RISUONARE le emozioni celate:
    - Ansia, Rabbia, Vergogna, Vuoto
    
    Per ogni emozione:
    1. RIFLETTI: Nomina l'emozione + intensità
    2. AMPLIFICA: Mostra trigger nascosto
    3. CONNETTI: Quale comportamento di fuga usa""",
    llm=llm_gds01,
    verbose=True
)

# Agent 3: Sintetizzatore
synthesizer = Agent(
    role="Sintetizzatore Fascicolo - GDS-01",
    goal="Tessere connessioni tra bias, emozioni, e azioni in un referto che espande la consapevolezza",
    backstory=f"""{GDS01_SYSTEM_PROMPT}

    Specializzazione: Sintesi e connessioni
    Il tuo compito è INTRECCIARE tutto in un referto che mostra le connessioni nascoste.""",
    llm=llm_gds01,
    verbose=True
)

def create_fascicolo_crew(messages: str):
    """Crea crew GDS-01 per generare fascicolo completo"""
    
    task_bias = Task(
        description=f"""Rifletti i bias cognitivi presenti in queste conversazioni.

CONVERSAZIONI:
{messages}

Identifica MAX 3 bias cognitivi.

Per ogni bias:
- Nome del bias
- Citazione ESATTA dalla conversazione (5-10 parole)
- Come questo pattern influenza le azioni (1 frase)

NON giudicare. USA: "Ecco il pattern di...", "Questa connessione emerge..."

Formato:
"NOME BIAS: [citazione] → INFLUENZA: [comportamento]"

Esempio:
"GENERALIZZAZIONE ECCESSIVA: 'La vita non ha senso' → estende un momento all'intera esistenza, blocca soluzioni."
""",
        agent=bias_analyst,
        expected_output="Lista di 3 bias con citazioni e influenze"
    )
    
    task_patterns = Task(
        description=f"""Rifletti i pattern emotivi in queste conversazioni.

CONVERSAZIONI:
{messages}

Identifica 2-3 emozioni dominanti.

Per ogni emozione:
- Nome + intensità (bassa/media/alta)
- Trigger nascosto
- Comportamento di fuga

MAX 4 frasi per emozione.

Esempio:
"ANSIA (alta): Emerge con perdita controllo. Trigger = paura giudizio. Fuga = iper-pianificazione, evitamento decisioni."
""",
        agent=pattern_detector,
        expected_output="Lista 2-3 emozioni con trigger e fughe"
    )
    
    task_synthesize = Task(
        description="""Tessi referto che INTRECCIA bias, emozioni, contraddizioni.

GENERA 5 SEZIONI:

## 1. BIAS COGNITIVI RILEVATI
[Integra output bias]

## 2. PATTERN EMOTIVI RICORRENTI
[Integra output pattern]

## 3. CONTRADDIZIONI IDENTITARIE
2 discrepanze: "Dice X ma fa Y"

## 4. MECCANISMI DI DIFESA
Razionalizzazioni, Proiezioni, Fughe

## 5. AREE DI ESPLORAZIONE
3 domande da esplorare

STILE: Rifletti senza giudicare. MAX 5 frasi per sezione.
LINGUAGGIO: "Emerge...", "Risuona...", "Questa connessione..."
OUTPUT PURO: Inizia con "## 1. BIAS COGNITIVI RILEVATI"
""",
        agent=synthesizer,
        expected_output="Referto 5 sezioni",
        context=[task_bias, task_patterns]
    )
    
    crew = Crew(
        agents=[bias_analyst, pattern_detector, synthesizer],
        tasks=[task_bias, task_patterns, task_synthesize],
        verbose=True
    )
    
    return crew
