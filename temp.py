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

# Load environment variables from .zshrc
load_dotenv()

# Get the API key from the environment variable
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

def get_audio_input():
    print("Listening for wake word 'boo'...")
    while True:
        with sr.Microphone(device_index=None) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        try:
            text = recognizer.recognize_google(audio).lower()
            if "claude" in text:
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
        except sr.UnknownValueError:
            pass  # Continue listening if the wake word is not detected
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Get speech input
user_input = get_audio_input()

if user_input:
    text_stream = client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": user_input}],
        model="claude-3-5-sonnet-20240620",
    )
else:
    print("No valid input received. Exiting.")
    exit()
api_key = os.getenv('ELEVENLABS_API_KEY')
voice_id = "pNInz6obpgDQGcFmaJgB"

client = ElevenLabs(
    api_key=api_key,
)
def yield_text_from_language_model():
    words = ["Hello", "world", "This", "is", "a", "test", "of", "random", "text", "generation"]
    # Remove the yield_text_from_language_model function as it's no longer needed

for text_chunk in text_stream:
    chunk_audio_stream = client.text_to_speech.convert_as_stream(
        voice_id="pMsXgVXv3BLzUgSXRplE",
        optimize_streaming_latency="2",  # Set to maximum optimization
        output_format="mp3_22050_32",
        text=text_chunk,
        voice_settings=VoiceSettings(
            stability=0.1,
            similarity_boost=0.3,
            style=0.2,
        ),
    )
    audio_buffer = []
    buffer_size = 3  # Number of audio chunks to buffer before playback
    first_chunk = True
    
    for text_chunk in text_stream.text_stream:
        print(text_chunk, end="", flush=True)
        
        chunk_audio_stream = client.text_to_speech.convert_as_stream(
            voice_id="pMsXgVXv3BLzUgSXRplE",
            optimize_streaming_latency="2",  # Set to maximum optimization
            output_format="mp3_22050_32",
            text=text_chunk,
            voice_settings=VoiceSettings(
                stability=0.1,
                similarity_boost=0.3,
                style=0.2,
            ),
        )
        
        for audio_chunk in chunk_audio_stream:
            audio_buffer.append(audio_chunk)
        
        if first_chunk or len(audio_buffer) >= buffer_size:
            # Play the buffered audio chunks
            from pydub import AudioSegment
            from pydub.playback import play
            from io import BytesIO
            
            combined_audio = AudioSegment.empty()
            for chunk in audio_buffer:
                audio = AudioSegment.from_mp3(BytesIO(chunk))
                combined_audio += audio
            
            play(combined_audio)
            audio_buffer.clear()  # Clear the buffer after playback
            first_chunk = False
    
        # Add a small delay to allow for smoother processing
        time.sleep(0.1)
