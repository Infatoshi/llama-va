import subprocess
import sys

def set_timer(hours=0, minutes=0, seconds=0):
    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    # AppleScript to create and start a timer
    apple_script = f'''
    tell application "Clock"
        activate
        tell application "System Events"
            delay 0.5
            click menu item "Timer" of menu "View" of menu bar 1  # Switch to Timer tab
            delay 0.5
            keystroke "t" using {{command down}}  # New Timer
            delay 0.5
            keystroke "{total_seconds}"  # Enter seconds
            delay 0.5
            keystroke return  # Start timer
            delay 0.5
            click button 2 of window 1  # Click the Start button (using button index)
        end tell
    end tell
    '''
    
    try:
        subprocess.run(['osascript', '-e', apple_script])
        print(f"Timer set for {hours}h {minutes}m {seconds}s")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        hours = int(sys.argv[1])
        minutes = int(sys.argv[2])
        seconds = int(sys.argv[3])
        set_timer(hours, minutes, seconds)
    else:
        print("Usage: python timer.py <hours> <minutes> <seconds>")
        print("Example: python timer.py 0 1 30 (for 1 minute and 30 seconds)")