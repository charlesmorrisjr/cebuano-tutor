import os
import json
import base64
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import edge_tts
import tempfile
from faster_whisper import WhisperModel
from data.scenarios import SCENARIOS

# Load environment variables securely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Define Models
class Correction(BaseModel):
    user_text: str
    corrected_text: str
    explanation: str

class WordDefinition(BaseModel):
    word: str
    definition: str

class TutorResponse(BaseModel):
    reply_cebuano: str
    reply_english: str
    glossary: list[WordDefinition]
    mistakes_found: bool
    corrections: list[Correction]

class UserMessage(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str
    voice: str | None = None

class StartRequest(BaseModel):
    scenario_id: str | None = None

# FastAPI App
app = FastAPI(title="Manang Tess Cebuano Tutor")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize GenAI Client
client = genai.Client()

def get_tutor_config(scenario_id=None):
    base_instruction = (
        "You are a warm, encouraging Cebuano language tutor named Manang Tess. "
        "The user is a beginner. Respond primarily in simple Metro Cebu Bisaya. "
        "Always evaluate the user's last message for grammatical, lexical, and focus-affix errors."
        "Pay special attention to Austronesian focus-affix errors (mo-, nag-, gi-, -on, i-). "
        "You must return your response strictly matching the required JSON schema. The explanation in class Correction should explain the error and the correction to an English speaker who is learning Cebuano."
        "Also, provide a glossary list for EVERY SINGLE WORD you use in reply_cebuano. You MUST NOT skip any words; even small connecting words, pronouns, or particles must be included. Each item in glossary must have 'word' (lowercase word without punctuation) and 'definition' (short English definition, e.g. word='maayong', definition='good')."
    )
    if scenario_id and scenario_id in SCENARIOS:
        scenario = SCENARIOS[scenario_id]
        scenario_context = f"\n\nCURRENT SCENARIO: {scenario['title_eng']}\n" \
                           f"Setting: {scenario['setting']}\n" \
                           f"Your Role: {scenario['tutor_role']}\n" \
                           f"User Role: {scenario['user_role']}\n" \
                           f"Goal: {scenario['goal']}\n" \
                           f"Key Vocabulary to reinforce: {', '.join(scenario['key_vocab'])}"
        base_instruction += scenario_context
        
    return types.GenerateContentConfig(
        system_instruction=base_instruction,
        temperature=0.3,
        response_mime_type="application/json",
        response_schema=TutorResponse,
    )

current_scenario_id = None

def format_tutor_response(data: dict) -> dict:
    """Helper to convert glossary list to dictionary for frontend lookups."""
    if isinstance(data.get("glossary"), list):
        data["glossary"] = {
            item["word"].lower().strip(): item["definition"]
            for item in data["glossary"]
            if isinstance(item, dict) and "word" in item and "definition" in item
        }
    return data

# Global chat session (single-user design)
chat_session = None

# Lazy-loaded Whisper Model
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (small)...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper model loaded!")
    return whisper_model

async def generate_audio_base64(text: str, voice: str = "fil-PH-BlessicaNeural") -> str:
    """Generates audio using edge-tts and returns it as a Base64 data URL."""
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    return f"data:audio/mp3;base64,{b64_audio}"

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/api/scenarios")
async def get_scenarios():
    return list(SCENARIOS.values())

@app.post("/api/start")
async def start_lesson(req: StartRequest):
    global chat_session, current_scenario_id
    current_scenario_id = req.scenario_id
    
    chat_session = client.chats.create(
        model="gemini-3.5-flash-lite", # Falls back gracefully if using standard APIs
        config=get_tutor_config(req.scenario_id)
    )
    
    first_message = "Hello, I am ready to start my Cebuano lesson."
    if req.scenario_id and req.scenario_id in SCENARIOS:
        scenario = SCENARIOS[req.scenario_id]
        first_message = f"[{scenario['title_eng']}] {scenario['starter_prompt']} Reply ONLY in valid JSON matching the schema."
        
    response = await asyncio.to_thread(chat_session.send_message, first_message)
    tutor_data = format_tutor_response(json.loads(response.text))
    
    # Generate TTS audio
    audio_url = await generate_audio_base64(tutor_data['reply_cebuano'])
    tutor_data['audio_base64'] = audio_url
    
    return tutor_data

@app.post("/api/chat")
async def send_message(user_msg: UserMessage):
    global chat_session
    if not chat_session:
        # Failsafe if user refreshed the page and continued chatting
        chat_session = client.chats.create(
            model="gemini-3.5-flash-lite",
            config=get_tutor_config(current_scenario_id)
        )
        
    response = await asyncio.to_thread(chat_session.send_message, user_msg.message)
    tutor_data = format_tutor_response(json.loads(response.text))
    
    # Generate TTS audio
    audio_url = await generate_audio_base64(tutor_data['reply_cebuano'])
    tutor_data['audio_base64'] = audio_url
    
    return tutor_data

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    voice = req.voice if req.voice else "fil-PH-BlessicaNeural"
    audio_url = await generate_audio_base64(req.text, voice=voice)
    return {"audio_base64": audio_url}

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    model = get_whisper_model()
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        segments, _ = await asyncio.to_thread(
            model.transcribe,
            tmp_path,
            initial_prompt="Maayong adlaw! Kumusta ka? Gusto ko mokat-on og Bisaya."
        )
        text = "".join([segment.text for segment in segments]).strip()
        return {"text": text}
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
