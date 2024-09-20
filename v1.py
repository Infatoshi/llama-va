from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
import random
import time

import anthropic
from dotenv import load_dotenv
import os
import speech_recognition as sr
from groq import Groq
import pyaudio
import io
from pydub import AudioSegment

# Load environment variables from .zshrc
load_dotenv()

# Get the API key from the environment variable
# Get the API key from the environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the context window
initial_context = [
    {
        "role": "system",
        "content": "You are a helpful voice assistant named Llama. You will give responses optimal for speech output (short and not boring)"
    }
]
context_window = initial_context.copy()

def get_audio_input():
    print("Listening for wake word 'llama'...")
    with sr.Microphone(device_index=None) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio).lower()
        if "llama" in text:
            print("Wake word detected. What would you like to say?")
            with sr.Microphone(device_index=None) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                main_input = recognizer.recognize_google(audio)
                print(f"You said: {main_input}")
                return main_input
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None
        elif "restart" or "reset" in text:
            return "restart"
    except sr.UnknownValueError:
        return None  # Continue listening if the wake word is not detected
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

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

# Main conversation loop
try:
    while True:
        user_input = get_audio_input()
        if user_input:
            if user_input.lower() == "restart":
                print("Restarting the conversation...")
                context_window = initial_context.copy()
                response_text = "I just cleared the context window"
            else:
                context_window.append({"role": "user", "content": user_input})
                
                chat_completion = client.chat.completions.create(
                    messages=context_window,
                    model="llama3-70b-8192",
                    temperature=0.5,
                    max_tokens=1024,
                )
                
                response_text = chat_completion.choices[0].message.content
                context_window.append({"role": "assistant", "content": response_text})
                print(f"Assistant said: {response_text}")
            
            # Stream the TTS audio
            audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
                voice_id="pMsXgVXv3BLzUgSXRplE",
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=response_text,
                voice_settings=VoiceSettings(
                    stability=0.1,
                    similarity_boost=0.3,
                    style=0.2,
                ),
            )
            
            # Play the audio stream
            play_audio_stream(audio_stream)
            
            print(f"You said: {user_input}")
        else:
            print("No valid input received. Listening again...")

except KeyboardInterrupt:
    print("\nExiting the conversation loop.")
