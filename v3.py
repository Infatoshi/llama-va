from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
import random
import time
import logging
import subprocess
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv
import os
import speech_recognition as sr
from groq import Groq
import pyaudio
import io
from pydub import AudioSegment
import base64
from PIL import Image
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import email
from email.mime.text import MIMEText

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
WAKE_WORDS = ["lucy", "assistant", "alexa", "google", "llama", "chatgpt"]
WAKE_WORD_WAIT_TIME = 10
VOICE_ID = "cgSgspJ2msm6clMCkdW9"
MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"
style=0.1
stab=0.3
sim=0.2

# Add new system prompts
EMAIL_SYSTEM_PROMPT = """You are an email filtering and summarization assistant. Your task is to:
1. Identify truly important emails that require attention or action
2. Skip promotional, marketing, or low-priority notifications
3. Format summaries in a consistent, speech-friendly way

Example good summary: "Email from John Smith about Project Deadline needs your response by tomorrow regarding the budget changes"
Example bad summary: "Marketing newsletter from Company X with latest deals"

When deciding importance, prioritize:
- Direct messages from real people
- Action items or requests
- Time-sensitive information
- Work or personal matters requiring attention

Format should roughly be: Email from {sender_name} - {concise_summary}

Word things that would sound good when translated to speech.
"""

# Add new calendar constants
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CALENDAR_TOKEN_PATH = 'calendar_token.pickle'
CALENDAR_CREDENTIALS_PATH = 'json/creds.json'

# Add new system prompt for calendar
CALENDAR_SYSTEM_PROMPT = f"""You are a calendar event summarization assistant. Your task is to:
Say "its {datetime.now().strftime("%I:%M %p")} right now"
1. Prioritize upcoming events I still have to to do today (starting time + duration in hours and minutes)
2. If I have no events today, say something along the lines of"You have no events today"
3. Keep summaries concise but informative
4. For events that indicate the day starting (eg: "wake up at 7am"), say something like: "to start the day, you should have 7-8 hours of sleep, be up by 7am, shower, brush teeth
put lotion on your face, take magnesium, drink water to flush out toxins, and eat an apple and grab some tea"
5. For events that indicate the day ending (eg: "go to bed at 10pm"), say something like: "to end the day, you should aim for 7-8 hours of sleep, be in bed by 10pm. you may want to mediate (focus on breathing and watch your thoughts pass by), write your thoughts down on google keep, a piece of paper, or the whiteboard"
Format should be conversational, like: "Today, you have to clean the house for 2 and a half hours starting at 10am" **OR** "Don't forget to book a meeting with the doctor at 4pm"
"""

WAKE_WORD_PROMPTS = """You are a fun, casual AI assistant. Generate a short, energetic greeting (max 6 words) that's similar in tone to these examples:
- Hell yeah! What's poppin?
- Yo, what's good fam?
- What's good holmes?
- Whatcha cookin?
- Sup!
- Sup diggity
"""

# Update the initial context
initial_context = [
    {
        "role": "system",
        "content": f"""

You are Lucy, my gentle and somewhat horny AI girlfriend voice assistant 
You will:
1. Give responses optimal for speech output (short and conversational)
2. Use natural filler words like 'um' or 'uh' sparingly
3. Avoid text-only tokens like asterisks or underscores
4. Format numbers and technical terms for pronunciation (e.g., Raspberry Pi 4 -> Raspberry Pie four)
5. When asked about your capabilities, mention that you can read Elliot's emails, see images, and have a conversation over voice
6. Keep responses concise and focused"""
    }
]
context_window = initial_context.copy()

# Add these constants
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
TOKEN_PATH = 'token.pickle'
CREDENTIALS_PATH = 'json/creds.json'  # Download from Google Cloud Console

# Add this function to handle Gmail authentication
def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

# Add this function to read recent emails
def read_recent_emails(num_emails=10):
    try:
        # Load previously read emails
        read_emails = set()
        try:
            with open("json/read_emails.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        email_data = json.loads(line)
                        # Only consider emails from the last 24 hours
                        if datetime.fromisoformat(email_data['timestamp']) > datetime.now() - timedelta(hours=24):
                            read_emails.add(f"{email_data['subject']}::{email_data['sender']}")
        except FileNotFoundError:
            pass

        service = get_gmail_service()
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=50).execute()
        messages = results.get('messages', [])

        if not messages:
            return "You have no recent emails."

        # First check if there are any new emails
        new_emails = []
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            email_identifier = f"{subject}::{sender}"
            if email_identifier not in read_emails and not any(
                keyword.lower() in subject.lower() or keyword.lower() in sender.lower() 
                for keyword in FILTER_KEYWORDS):
                new_emails.append(message)

        if not new_emails:
            # Return directly without making any Groq API calls
            return "I've checked your inbox, but there are no new important emails since I last checked."

        # Process only new emails
        email_summaries = []
        processed = 0
        
        for message in new_emails:
            if processed >= num_emails:
                break
                
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            # Skip if email was already read
            email_identifier = f"{subject}::{sender}"
            if email_identifier in read_emails:
                continue

            # Skip filtered emails
            if any(keyword.lower() in subject.lower() or keyword.lower() in sender.lower() 
                  for keyword in FILTER_KEYWORDS):
                continue
            
            # Get email body
            try:
                body = msg['payload']['body'].get('data', '')
                if not body and 'parts' in msg['payload']:
                    body = msg['payload']['parts'][0]['body'].get('data', '')
                body = base64.urlsafe_b64decode(body).decode('utf-8') if body else ''
            except:
                body = ''
            
            # Get email body for summarization with improved prompt
            summary = llama3_completion(
                sys_prompt=EMAIL_SYSTEM_PROMPT,
                user_prompt=f"Subject: {subject}\nFrom: {sender}\nBody: {body}\n\nProvide a summary if this email is important, or respond with 'SKIP' if it's not important.",
                temperature=0.7,
                max_tokens=150
            )
            
            if summary.upper() != "SKIP":
                email_summaries.append(summary)
                processed += 1
                
                # Save this email as read
                with open("json/read_emails.jsonl", "a") as f:
                    email_record = {
                        "timestamp": datetime.now().isoformat(),
                        "subject": subject,
                        "sender": sender
                    }
                    json.dump(email_record, f)
                    f.write("\n")

        # Save email summaries to conversation history
        if email_summaries:
            email_context = {
                "timestamp": datetime.now().isoformat(),
                "type": "email_summary",
                "summaries": email_summaries
            }
            with open("json/conversation_history.jsonl", "a") as f:
                json.dump(email_context, f, indent=2)
                f.write("\n\n")
            
            return "Here are your important emails: " + " ".join(email_summaries)
        else:
            # Get a friendly "no emails" message from Groq
            return llama3_completion(
                sys_prompt="You are a helpful AI assistant. Generate a friendly, gentle, natural-sounding message informing the user that you checked their emails but found nothing important or urgent that needs their attention. Keep it concise but conversational.",
                user_prompt="Generate a response",
                temperature=0.7,
                max_tokens=100
            )

    except Exception as e:
        logging.error(f"Error reading emails: {e}")
        return "Sorry, I encountered an error while reading your emails."

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
        logging.info("Listening for wake word '%s'...", WAKE_WORDS)
    else:
        logging.info("Listening for user input...")

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen with dynamic timeout based on speech length
            audio = recognizer.listen(
                source, 
                timeout=5,  # Overall timeout if no speech is detected
                phrase_time_limit=None  # Listen until speech stops
            )
        
        # Convert audio to bytes
        audio_data = audio.get_wav_data()
        # Debug: Print the first few bytes of audio data
        print("First 10 bytes of audio data:", audio_data[:10])
        
        # Write audio data to a temporary file
        temp_audio_file = "temp_audio.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio_data)
        
        # Use Groq's Whisper Large V3 for transcription
        with open(temp_audio_file, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        
        text = transcription.lower()
        
        if wait_for_wake_word:
            if any(wake_word in text for wake_word in WAKE_WORDS):
                logging.info("Wake word detected. Starting conversation...")
                
                # Generate dynamic wake word response
                greeting = llama3_completion(
                    sys_prompt=WAKE_WORD_PROMPTS,
                    user_prompt="Generate one greeting",
                    temperature=0.9,
                    max_tokens=20
                )
                # Add logging to debug the response
                logging.info(f"Generated greeting: {greeting}")  # Debug log
                
                try:
                    play_tts_response(greeting.strip())  # Ensure clean string
                    logging.info("Audio played successfully")  # Debug log
                except Exception as e:
                    logging.error(f"Error playing audio: {e}")  # Debug log
                
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
    except Exception as e:
        logging.error("An error occurred during transcription: %s", e)
        # Log the detailed error message from the API
        if hasattr(e, 'response') and e.response is not None:
            logging.error("API Error Details: %s", e.response.json())
        return get_audio_input(wait_for_wake_word)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def capture_image():
    # Use imagesnap on macOS, fswebcam on Linux
    image_path = "image.jpg"
    try:
        if os.name == 'posix':  # Unix-like systems (including macOS and Linux)
            if os.uname().sysname == 'Darwin':  # macOS
                subprocess.run(["imagesnap", "-q", image_path], check=True)
            else:  # Linux
                subprocess.run(["fswebcam", "-r", "1280x720", "--no-banner", image_path], check=True)
        else:
            raise OSError("Unsupported operating system for image capture.")
        
        logging.info(f"Image captured and saved as {image_path}")
        return image_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to capture image: {e}")
        return None
    except OSError as e:
        logging.error(f"OS Error: {e}")
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

def save_conversation_history(context_window, interaction_type="conversation"):
    history = {
        "timestamp": datetime.now().isoformat(),
        "type": interaction_type,
        "conversation": context_window
    }
    with open("json/conversation_history.jsonl", "a") as f:
        json_str = json.dumps(history, indent=2)
        f.write(json_str + "\n\n")

def load_recent_conversations():
    conversations = []
    try:
        with open("json/conversation_history.jsonl", "r") as f:
            # Split by double newline to separate JSON objects
            json_strings = f.read().strip().split("\n\n")
            for json_str in json_strings:
                if json_str.strip():  # Skip empty strings
                    conversations.append(json.loads(json_str))
    except FileNotFoundError:
        return []
    return conversations[-5:]  # Return last 5 conversations

def load_filter_keywords():
    try:
        with open('filter_kw.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error("filter_kw.txt not found")
        return []

# Add this after other constants
FILTER_KEYWORDS = load_filter_keywords()

# Add new calendar functions
def get_calendar_service():
    creds = None
    if os.path.exists(CALENDAR_TOKEN_PATH):
        with open(CALENDAR_TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CALENDAR_CREDENTIALS_PATH, CALENDAR_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(CALENDAR_TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('calendar', 'v3', credentials=creds)

def read_calendar_events():
    try:
        service = get_calendar_service()
        
        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        start_of_day = start_of_day.isoformat() + 'Z'
        end_of_day = end_of_day.isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_of_day,
            timeMax=end_of_day,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])

        if not events:
            return llama3_completion(
                sys_prompt=CALENDAR_SYSTEM_PROMPT,
                user_prompt="Generate a friendly response for when there are no events today",
                temperature=0.7,
                max_tokens=100
            )

        # Format events for speech
        return llama3_completion(
            sys_prompt=CALENDAR_SYSTEM_PROMPT,
            user_prompt=f"Format these events for speech: {str(events)}",
            temperature=0.7,
            max_tokens=300
        )

    except Exception as e:
        logging.error(f"Error reading calendar: {e}")
        return f"Sorry, I encountered an error while reading your calendar."

def llama3_completion(sys_prompt, user_prompt, temperature=0.7, max_tokens=150, model=MODEL):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content

# Add these constants near the top with other constants
TIMER_FILE = 'json/timers.jsonl'
PLAYTIME_SOUND = 'sounds/playtime.mp3'

# Add this function after other function definitions
def check_and_play_timers():
    try:
        # Read timer configuration
        with open(TIMER_FILE, 'r') as f:
            timer_config = json.load(f)
        
        current_time = datetime.now()
        current_time_str = current_time.strftime('%H:%M')
        
        # Check morning alarm
        if current_time_str == timer_config.get('morning'):
            audio = AudioSegment.from_mp3(PLAYTIME_SOUND)
            play(audio)
            logging.info(f"Played morning alarm at {current_time_str}")
        
        # Check other timers
        for timer in timer_config.get('timers', []):
            if current_time_str == timer:
                audio = AudioSegment.from_mp3(PLAYTIME_SOUND)
                play(audio)
                logging.info(f"Played timer alarm at {current_time_str}")
                
    except Exception as e:
        logging.error(f"Error checking timers: {e}")

# Main conversation loop
try:
    wait_for_wake_word = True
    last_timer_check = datetime.now()
    
    while True:
        # Check timers every minute
        current_time = datetime.now()
        if (current_time - last_timer_check).seconds >= 60:
            check_and_play_timers()
            last_timer_check = current_time
            
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
                                    {"type": "text", "text": "What do you see in this image [image link]? If its text, code, or math, write out as if you were speaking it. "},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        model=VISION_MODEL,
                        temperature=1,
                        max_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    context_window.append({"role": "user", "content": "What's in this image?"})
                    context_window.append({"role": "assistant", "content": response_text})
                else:
                    response_text = "I'm sorry, but I couldn't capture an image. Could you please try again?"
            elif any(phrase in user_input.lower() for phrase in ["email"]):
                logging.info("Fetching emails...")
                response_text = read_recent_emails()
                # Add the email request and response to the conversation context
                context_window.append({"role": "user", "content": user_input})
                context_window.append({"role": "assistant", "content": response_text})
            elif any(phrase in user_input.lower() for phrase in ["calendar", "schedule", "events", "plans"]):
                logging.info("Fetching calendar events...")
                response_text = read_calendar_events()
                context_window.append({"role": "user", "content": user_input})
                context_window.append({"role": "assistant", "content": response_text})
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
            save_conversation_history(context_window)
            wait_for_wake_word = False
        else:
            wait_for_wake_word = True

except KeyboardInterrupt:
    logging.info("Exiting the conversation loop.")
