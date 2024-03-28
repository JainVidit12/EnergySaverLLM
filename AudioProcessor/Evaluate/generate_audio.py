from pathlib import Path
import openai

import json
import random
import ffmpeg
import soundfile as sf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# /scr1/vjain018/EnergySaverLLM/AudioProcessor/Evaluate/EV_combined.benchmark.json
path_prefix = '/scr1/vjain018/EnergySaverLLM/AudioProcessor/Evaluate/'
input_benchmark_filename = path_prefix + 'EV_combined.benchmark.json'

f = open(input_benchmark_filename)

input_text_benchmark = json.load(f)

voices = ['alloy','echo','shimmer','fable','onyx','nova']



for sample in input_text_benchmark:
    if 'end_charge_time' not in sample['json_str']:
        continue

    text = sample['prompt']
    index = sample['index']

    voice = voices[index%len(voices)]

    speech_file_path_text = path_prefix + 'audio_samples/'+str(index)+'_'+voice+'.wav'
    speech_file_path_object = Path(speech_file_path_text)

    response = openai.audio.speech.create(
                model = "tts-1",
                voice = voice,
                input = text,
                response_format = 'wav'
                )
    
    response.stream_to_file(speech_file_path_object)

    # stream = ffmpeg.input(speech_file_path_text)
    # stream = ffmpeg.output(stream, filename = speech_file_path_text, **{'ar': '16000','acodec':'flac'})
    # ffmpeg.run(stream)
    # break