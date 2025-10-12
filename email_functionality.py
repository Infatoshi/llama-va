from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Gmail API settings
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
TOKEN_PATH = 'token.pickle'
CREDENTIALS_PATH = 'creds.json'

def load_filter_keywords():
    try:
        with open('filter_kw.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error("filter_kw.txt not found")
        return []

# Replace the FILTER_KEYWORDS constant with a function call
FILTER_KEYWORDS = load_filter_keywords()

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

def read_recent_emails(num_emails=10):
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=num_emails).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No recent emails found."

        email_summaries = []
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            # Skip this email if it contains any filtered keywords
            if any(keyword.lower() in subject.lower() or keyword.lower() in sender.lower() 
                  for keyword in FILTER_KEYWORDS):
                continue
            
            email_summaries.append(f"\nFrom: {sender}\nSubject: {subject}\n")

        return "\n---".join(email_summaries)

    except Exception as e:
        logging.error(f"Error reading emails: {e}")
        return f"Error reading emails: {str(e)}"

def main():
    print("Fetching your recent emails...\n")
    emails = read_recent_emails()
    print(emails)

if __name__ == "__main__":
    main() 