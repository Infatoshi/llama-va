from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Calendar API settings
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
TOKEN_PATH = 'calendar_token.pickle'
CREDENTIALS_PATH = 'creds.json'

def get_calendar_service():
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
    
    return build('calendar', 'v3', credentials=creds)

def read_calendar_events(days_ahead=1):
    try:
        service = get_calendar_service()
        
        # Get the start and end of today
        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Convert to RFC3339 format
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
            return "No events scheduled for today."

        event_summaries = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            
            # Format the date/time in a more readable way
            if 'T' in start:  # This is a datetime (not just a date)
                formatted_time = start_dt.strftime("%I:%M %p")  # Only show time for today's events
            else:
                formatted_time = "All day"
                
            summary = event.get('summary', 'Untitled Event')
            event_summaries.append(f"\nEvent: {summary}\nWhen: {formatted_time}\n")

        return f"Today's Schedule ({now.strftime('%B %d')}): \n---" + "\n---".join(event_summaries)

    except Exception as e:
        logging.error(f"Error reading calendar: {e}")
        return f"Error reading calendar: {str(e)}"

def main():
    print("Fetching your upcoming calendar events...\n")
    events = read_calendar_events()
    print(events)

if __name__ == "__main__":
    main() 