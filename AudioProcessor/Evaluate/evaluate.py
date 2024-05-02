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



    def _downSample(data, sample_rate, target_sampling_rate = 16000):
        number_of_samples = round(len(data) * float(target_sampling_rate) / sample_rate)
        data = sps.resample(data, number_of_samples)
        return data

    def processAudio(self, path) -> str:
        

        data, sample_rate = sf.read(path)
        if sample_rate > 16000:
            data = self._downSample(data, sample_rate)
        input_features = self._processor(data, sampling_rate = sampling_rate, return_tensors="pt").input_features 
        predicted_ids = self._model.generate(input_features)
        transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription

path_prefix = '/scr1/vjain018/EnergySaverLLM/AudioProcessor/Evaluate/'
audioprocessor = AudioProcessor()

whisper  = pipeline("automatic-speech-recognition",
                    "openai/whisper-base",
                    device="cuda:0")

if __name__ == "__main__":
    
    input_benchmark_filename = path_prefix + 'benchmark.json'
    f = open(input_benchmark_filename)
    input_text_benchmark = json.load(f)

    sound_samples = os.listdir('audio_samples/')

    logs = []
    target_sampling_rate = 16000
    for sample in sound_samples:
        number = int(sample.split('_')[0])
        # print(path_prefix+'audio_samples/'+sample)
        
        data, sample_rate = sf.read(path_prefix+'audio_samples/'+sample)

        number_of_samples = round(len(data) * float(target_sampling_rate) / sample_rate)
        data = sps.resample(data, number_of_samples)
        
        text_from_audio = audioprocessor.processAudio(data, target_sampling_rate)[0]
        # print(text_from_audio)
        # text_from_audio = transcription = whisper('audio_samples/'+sample,
        #                 chunk_length_s=30)
        
        sample_dict = input_text_benchmark[number]
        ground_truth = sample_dict['prompt']
        ground_truth_json = sample_dict['json']


        logs.append({
            'predicted':text_from_audio,
            'actual':ground_truth,
            'index':number,
            'json':ground_truth_json,
        })
        
    with open('result_audio.json', 'w') as f:
        json.dump(logs, f, indent=4)
