from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
import random
import time
import logging
import subprocess

import anthropic
from dotenv import load_dotenv
import os
import speech_recognition as sr
from groq import Groq
import pyaudio
import io
from pydub import AudioSegment
import base64
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .zshrc
load_dotenv()

# Get the API key from the environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Constants
WAKE_WORD = "lucy"
WAKE_WORD_WAIT_TIME = 10
VOICE_ID = "cgSgspJ2msm6clMCkdW9"
MODEL = "llama3-70b-8192"
style=0.1
stab=0.3
sim=0.2

# Initialize the context window
initial_context = [
    {
        "role": "system",
        "content": f"You are a helpful voice assistant named {WAKE_WORD}. You will give responses optimal for speech output (short and with conversational characters/words like 'um' or 'uh', and without text-only tokens like asterisks, underscores, etc). Ensure the output is designed for pronunciation (not text). Example: Raspberry Pi 4 -> Raspberry Pie four. "
    }
]
context_window = initial_context.copy()

def play_audio_stream(audio_stream):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Collect all chunks into a single bytes object
    audio_data = b''.join(chunk for chunk in audio_stream)

    # Convert MP3 to raw PCM audio
    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    raw_data = audio.raw_data

    # Open a stream
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels,
                    rate=audio.frame_rate,
                    output=True)

    # Play the audio
    chunk_size = 1024
    offset = 0
    while offset < len(raw_data):
        chunk = raw_data[offset:offset + chunk_size]
        stream.write(chunk)
        offset += chunk_size

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()

def get_audio_input(wait_for_wake_word=True):
    if wait_for_wake_word:
        logging.info("Listening for wake word '%s'...", WAKE_WORD)
    else:
        logging.info("Listening for user input...")

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        
        text = recognizer.recognize_google(audio).lower()
        
        if wait_for_wake_word:
            if WAKE_WORD in text:
                logging.info("Wake word detected. Starting conversation...")
                play_tts_response("Hell yeah! Whats poppin?")
                return get_audio_input(wait_for_wake_word=False)
            elif "restart" in text or "reset" in text:
                return "restart"
        else:
            logging.info("User said: %s", text)
            if "look at" in text or "what do you see" in text or "picture" in text or "image" in text or "photo" in text:
                return "vision_request" + text
            return text
    except sr.WaitTimeoutError:
        logging.warning("Listening timed out. Reverting to wake word mode.")
        return None
    except sr.UnknownValueError:
        logging.warning("Didn't catch that. Please try again.")
        return get_audio_input(wait_for_wake_word)
    except sr.RequestError as e:
        logging.error("Could not request results from Google Speech Recognition service: %s", e)
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def capture_image():
    # Use fswebcam to capture an image
    image_path = "image.jpg"
    try:
        subprocess.run(["fswebcam", "-r", "1280x720", "--no-banner", image_path], check=True)
        logging.info(f"Image captured and saved as {image_path}")
        return image_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to capture image: {e}")
        return None

def play_tts_response(text):
    audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
        voice_id=VOICE_ID,
        optimize_streaming_latency="2",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=stab,
            similarity_boost=sim,
            style=style,
        ),
    )
    play_audio_stream(audio_stream)

# Main conversation loop
try:
    wait_for_wake_word = True
    while True:
        user_input = get_audio_input(wait_for_wake_word)
        if user_input:
            if user_input.lower() == "restart":
                logging.info("Restarting the conversation...")
                context_window = [context_window[0]]
                response_text = "I just cleared the context window"
                wait_for_wake_word = True
            elif "vision_request" in user_input:
                image_path = capture_image()
                if image_path:
                    base64_image = encode_image(image_path)
                    
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"What do you see in this image [image link]? If its text, code, or math, write out as if you were speaking it. Keep this prompt in mind about the image {user_input[14:]}"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        model="llava-v1.5-7b-4096-preview",
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    context_window.append({"role": "user", "content":f"What do you see in this image [image link]? If its text, code, or math, write out as if you were speaking it. Keep this prompt in mind about the image {user_input[14:]}"})
                    context_window.append({"role": "assistant", "content": response_text})
                else:
                    response_text = "I'm sorry, but I couldn't capture an image. Could you please try again?"
            else:
                context_window.append({"role": "user", "content": user_input})
                
                chat_completion = client.chat.completions.create(
                    messages=context_window,
                    model=MODEL,
                    temperature=0.6,
                    max_tokens=1024,
                )
                
                response_text = chat_completion.choices[0].message.content
                context_window.append({"role": "assistant", "content": response_text})
                logging.info("Assistant said: %s", response_text)
            
            play_tts_response(response_text)
            wait_for_wake_word = False
        else:
            wait_for_wake_word = True

except KeyboardInterrupt:
    logging.info("Exiting the conversation loop.")
