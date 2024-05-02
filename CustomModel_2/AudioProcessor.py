from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

from typing import Optional
import soundfile as sf
import glob
import os
import json
import numpy as np

from scipy.io import wavfile
import scipy.signal as sps


class AudioProcessor():
    
    def __init__(
        self,
        model_name : Optional[str] = "openai/whisper-base",
        gpu_no : Optional[int] = None
    ):
        self._processor = WhisperProcessor.from_pretrained(model_name)

        if gpu_no is None:
            self._model = WhisperForConditionalGeneration.from_pretrained(model_name)
        else:
            self._model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map = gpu_no)
        self._model.config.forced_decoder_ids = None

        self._device = torch.device('cuda:'+str(gpu_no))
        self._gpu_no = gpu_no



    def _downSample(self, audio_array, sample_rate, target_sampling_rate = 16000):
        number_of_samples = round(len(audio_array) * float(target_sampling_rate) / sample_rate)
        audio_array = sps.resample(audio_array, number_of_samples)
        return audio_array

    def processAudio(self, path) -> str:
        

        audio_content, sample_rate = sf.read(path)
        
        if sample_rate > 16000:
            audio_content = self._downSample(audio_content, sample_rate)
        
        with torch.cuda.device(self._gpu_no):
            input_features = self._processor(audio_content, sampling_rate = 16000, return_tensors="pt").input_features.to(device = self._device)
            predicted_ids = self._model.generate(input_features)
            transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True, language = 'en')
        # print(transcription)
        print("transcription: " + transcription[0])

        return transcription[0]
