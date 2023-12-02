# from transformers import GenerationConfig

# # Download configuration from huggingface.co and cache.
# generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
# print(generation_config.__dict__.keys())

import whisperx

device = "cuda"
audio_file = "/content/vi_vocals (1) (mp3cut.net).wav"

modelx = whisperx.load_model("large-v3", device, compute_type="float32", language='vi') #medium
print('model loaded')

audio = whisperx.load_audio(audio_file)
print('audio loaded')

data = modelx.transcribe(audio, batch_size=1)

print(data)




# # import librosa
# # import numpy as np
# # from onnxruntime import InferenceSession
# # import os
# # import subprocess
# # import time
# # from transformers import WhisperProcessor

# # N_FRAMES = 3000
# # HOP_LENGTH = 160
# # SAMPLE_RATE = 16000
# # N_MELS = 80

# # min_length = 0
# max_length = 100
# # repetition_penalty = 1.0

# audio = librosa.load("/content/vi_vocals (1) (mp3cut.net).wav")[0]

# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language='vi', task="transcribe")
# inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
# input_features = inputs.input_features

# onnx_path='/content/whisper-large-v3/whisper-large-v3_beamsearch.onnx'
# # sess = InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

# beam_size = 5
# NUM_RETURN_SEQUENCES = 1
# input_shape = [1, N_MELS, N_FRAMES]

# ort_inputs = {
#     "input_features": np.array(features, dtype=np.float32),
#     "max_length": np.array([max_length], dtype=np.int32),
#     "min_length": np.array([min_length], dtype=np.int32),
#     "num_beams": np.array([beam_size], dtype=np.int32),
#     "num_return_sequences": np.array([NUM_RETURN_SEQUENCES], dtype=np.int32),
#     "length_penalty": np.array([1.0], dtype=np.float32),
#     "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
#     "decoder_input_ids": np.array([[50258, 50278, 50360, 50364]], dtype=np.int32) ,
#     # "decoder_input_ids": np.array([[50258, 50364, 50258, 50278, 50360, 50364, 50257]], dtype=np.int32) ,
#     # "attention_mask": np.zeros(input_shape).astype(np.int32),
# }

# out = sess.run(None, ort_inputs)[0]
# transcription = processor.batch_decode(out[0], skip_special_tokens=True)[0]
# print(transcription)

# # Timed run
# start = time.time()
# for i in range(10):
#     _ = sess.run(None, ort_inputs)
# diff = time.time() - start
# print(f"time {diff/10} sec")

