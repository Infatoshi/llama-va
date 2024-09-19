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
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Get speech input
user_input = get_audio_input()

if user_input:
    with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": user_input}],
        model="claude-3-5-sonnet-20240620",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
else:
    print("No valid input received. Exiting.")
