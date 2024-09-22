## Overview
- a voice assistant using Groq's llama3.1:70b w/ super fast inference in combination with elevenlabs API for TTS
- uses google SR for STT
- uses pyAudio for audio output

> an upgraded version of [this](https://github.com/Infatoshi/chatgpt-voice-assistant)

## Installation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install portaudio19-dev clang pulseaudio alsa-utils alsa-tools libasound2-dev flac
pip install -r requirements.txt
```
## TODO
- record conversations in JSON for future RLHF
- add a GUI w/ 3d avatars (ai waifus)
- find ways to decrease response time latency
- training a custom voice model to sound like certain people
- integrate with microcontrollers to automate home appliances
- lock doors, turn on lights, etc.

