import whisperx

audio_file = "/content/htdemus-export/output/quan-hoc_vocals.wav"

modelx = whisperx.load_model()
print('model loaded')

audio = whisperx.load_audio(audio_file)
print('audio loaded')

data = modelx.transcribe(audio, batch_size=1)

print(data)




