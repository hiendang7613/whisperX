# from transformers import GenerationConfig

# # Download configuration from huggingface.co and cache.
# generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
# print(generation_config.__dict__.keys())

import whisperx

device = "cpu"
audio_file = "path"

modelx = whisperx.load_model()
print('model loaded')

audio = whisperx.load_audio(audio_file)
print('audio loaded')

data = modelx.transcribe(audio, batch_size=1)

print(data)




