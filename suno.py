import torch
import sounddevice as sd
from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

# Enable CPU offloading for memory efficiency
model.enable_cpu_offload()

# Convert to BetterTransformer for speed
model = model.to_bettertransformer()
for _ in range(2):
# Prepare input
    text = "The only reason Elon is the richest man in the World is because he lives rent free in so many people’s heads."
    inputs = processor(text=text, return_tensors="pt", voice_preset="v2/en_speaker_9").to(device)

# Generate speech
    with torch.no_grad():
        speech_values = model.generate(**inputs, do_sample=True)

# Cast to float32 and move to CPU
    speech_values_float32 = speech_values.to(torch.float32).cpu()

# Normalize to [-1, 1] range
    speech_values_float32 = torch.clamp(speech_values_float32, -1, 1)

# Get the sampling rate
    sampling_rate = model.generation_config.sample_rate

# Convert to NumPy array for saving as WAV
    audio_data_np = speech_values_float32.numpy().squeeze()

# Save as WAV file (converting to int16 for WAV format)
    scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=(audio_data_np * 32767).astype('int16'))

# Play the audio
    sd.play(audio_data_np, sampling_rate)
    sd.wait()


