# from transformers import GenerationConfig

# # Download configuration from huggingface.co and cache.
# generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
# print(generation_config.__dict__.keys())

import whisperx

device = "cpu"
audio_file = "/Users/apple/Downloads/vi_vocals (1) (mp3cut.net).wav"

modelx = whisperx.load_model("large-v3", device, compute_type="float32", language='vi') #medium
print('model loaded')

audio = whisperx.load_audio(audio_file)
print('audio loaded')

data = modelx.transcribe(audio, batch_size=1)

print(data)










def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, model_fp=None):
    vad_model = Model.from_pretrained(model_fp, use_auth_token=None)
    hyperparameters = {"onset": vad_onset, 
                    "offset": vad_offset,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1}
    vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)
    return vad_pipeline


def load_model(whisper_arch,
               device,
               compute_type="float16",
               language ='vi',
               task="transcribe"):

    model = WhisperModel(whisper_arch,
                         device=device,
                         device_index=0,
                         compute_type=compute_type,
                         download_root=None,
                         cpu_threads=4)
    
    model_id = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
    }

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    # default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    vad_model = load_vad_model(torch.device(device), model_fp='/Users/apple/Downloads/whisperx-vad-segmentation.bin', **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )




def load_audio(file: str, sr: int = SAMPLE_RATE):
    try:
        cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


model = load_model(model_name, device=device, device_index=0, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=4)

audio = load_audio(audio_path)
# >> VAD & ASR

result = model.transcribe(audio, batch_size=1)

