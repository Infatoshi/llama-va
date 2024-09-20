import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Use mixed precision for faster computation
torch.set_float32_matmul_precision('medium')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer only once
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Pre-tokenize the description
description = "Laura takes her time while speaking. She is gentle with her words is expressive for punctation. No background noise."
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)

def generate_audio(prompt):
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr

# Example usage
prompt = "Master, I missed you! Can we have some fun now?"
audio_arr = generate_audio(prompt)
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

