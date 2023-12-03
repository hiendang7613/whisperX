import os
import warnings
from typing import List, Union, Optional, NamedTuple

# import ctranslate2
# import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

from transformers import GenerationConfig
from onnxruntime import InferenceSession
from transformers import WhisperProcessor
onnx_path='/content/whisper-large-v3_beamsearch.onnx'
generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
repetition_penalty=generation_config.repetition_penalty
sess = InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language='vi', task="transcribe")

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel:

    def generate_segment_batched(self, features, options):
        batch_size = features.shape[0]
        ort_inputs = {
            "input_features": np.array(features, dtype=np.float32),
            # "decoder_input_ids": np.array([[50360]] * batch_size, dtype=np.int32) ,
            "max_length": np.array([448], dtype=np.int32),
            "min_length": np.array([0], dtype=np.int32),
            "num_beams": np.array([5], dtype=np.int32),
            "num_return_sequences": np.array([1], dtype=np.int32),
            "length_penalty": np.array([1], dtype=np.float32),
            "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
            "decoder_input_ids": np.array([[50258, 50278, 50360, 50364]] * batch_size , dtype=np.int32) ,
            # "decoder_input_ids": np.array([[50258, 50364, 50258, 50278, 50360, 50364, 50257]]* batch_size, dtype=np.int32) ,
            # "attention_mask": np.zeros(input_shape).astype(np.int32),
        }

        out = sess.run(None, ort_inputs)[0]
        text = []
        for s in out:
          text.append(processor.batch_decode(s, skip_special_tokens=True))
        return text


class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            vad,
            vad_params: dict,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = 128
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language='vi', task='transcribe', chunk_size=30, print_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append({
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        return {"segments": segments, "language": language}


def load_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               task="transcribe",
               download_root=None,
               threads=4):

    # if whisper_arch.endswith(".en"):
    #     language = "en"

    model = WhisperModel()

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

    # if vad_options is not None:
    #     default_vad_options.update(vad_options)

    vad_model = load_vad_model(torch.device(device), 
      model_fp='/content/whisperx-vad-segmentation.bin',
     **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
