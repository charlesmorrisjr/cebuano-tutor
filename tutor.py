import os
import sys
import json
import ctypes
import subprocess
from google import genai
from google.genai import types
from pydantic import BaseModel

# Load environment variables from .env file
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

# 1. Define the strict data structure
class Correction(BaseModel):
    user_text: str
    corrected_text: str
    explanation: str

class TutorResponse(BaseModel):
    reply_cebuano: str
    reply_english: str
    mistakes_found: bool
    corrections: list[Correction]

# 2. Initialize the client (automatically reads GEMINI_API_KEY from .env)
client = genai.Client()

tutor_config = types.GenerateContentConfig(
    system_instruction=(
        "You are a warm, encouraging Cebuano language tutor named Manang Tess. "
        "The user is a beginner. Respond primarily in simple Metro Cebu Bisaya. "
        "Always evaluate the user's last message for grammatical, lexical, and focus-affix errors. "
        "You must return your response strictly matching the required JSON schema."
    ),
    temperature=0.3,
    response_mime_type="application/json",
    response_schema=TutorResponse,
)

def play_audio(text):
    """Generates ultra-fast TTS and plays it directly via the Windows kernel."""
    audio_file = os.path.abspath("response.mp3")
    
    # 1. Generate audio rapidly using Edge's Neural TTS
    subprocess.run([
        sys.executable, "-m", "edge_tts", 
        "--voice", "fil-PH-BlessicaNeural", 
        "--text", text, 
        "--write-media", audio_file
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Play the audio with zero startup latency using ctypes/winmm
    winmm = ctypes.windll.winmm
    winmm.mciSendStringW('close my_audio', None, 0, None)
    
    # Open the MP3 file natively
    open_cmd = f'open "{audio_file}" type mpegvideo alias my_audio'
    winmm.mciSendStringW(open_cmd, None, 0, None)
    
    # Play and pause script execution until the audio finishes
    winmm.mciSendStringW('play my_audio wait', None, 0, None)
    winmm.mciSendStringW('close my_audio', None, 0, None)

def main():
    print("Starting Optimized Cebuano Tutor... (Type 'quit' to exit)")
    print("-" * 50)
    
    chat = client.chats.create(
        model="gemini-3.5-flash-lite", 
        config=tutor_config
    )
    
    response = chat.send_message("Hello, I am ready to start my Cebuano lesson.")
    
    while True:
        tutor_data = json.loads(response.text)
        
        print(f"\n🗣️ Tutor (Cebuano): {tutor_data['reply_cebuano']}")
        print(f"📖 Translation: {tutor_data['reply_english']}")
        
        # Audio playback will now start almost immediately
        play_audio(tutor_data['reply_cebuano'])
        
        if tutor_data.get('mistakes_found') and tutor_data.get('corrections'):
            print("\n⚠️  CORRECTIONS:")
            for c in tutor_data['corrections']:
                print(f"  - You said: {c['user_text']}")
                print(f"  - Should be: {c['corrected_text']}")
                print(f"  - Why: {c['explanation']}")
        
        print("-" * 50)
        
        user_input = input("\nYou (Type in Cebuano): ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = chat.send_message(user_input)
        
    if os.path.exists("response.mp3"):
        try:
            os.remove("response.mp3")
        except PermissionError:
            pass

if __name__ == "__main__":
    main()