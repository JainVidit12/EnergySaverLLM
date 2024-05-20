# %% Importing stuff
from pathlib import Path
import openai
import random

# import auxillary packages
import json
from typing import Optional, List, Dict
import torch

from Agent import ChargingAgent, extract_code, reset_params_file, clear_param_backups, create_dict
from transformers import BitsAndBytesConfig

# global params_filepath
params_filepath = "params/EVCharging.json"
params_filepath_original = "params/EVCharging_original.json"

path_prefix = '/scr1/vjain018/EnergySaverLLM/CustomModel_2/evaluate/'
input_benchmark_filename = path_prefix + 'benchmark.json'
results_filename = path_prefix + 'results_noQ.json'


# %% Few Shot Prompt
example_qa = """
<|user|>
Instruction: Charge the car till 90%.
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_level": 0.9
```
<|end|>

<|user|>
Instruction: Charge the car till 2:15 AM.
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_time": "0215"
```
<|end|>

<|user|>
Question: Charge the car till 80% by 3:45 PM
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_level": 0.8,
"end_charge_time": "1545"
```
<|end|>

"""
fourBit_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
eightBit_config = BitsAndBytesConfig(load_in_8bit=True)



def generate_audio_file(
    message : str,
    path : 'temp.WAV'
    ):
    speech_file_path_object = Path(path)

    voices = ['alloy','echo','shimmer','fable','onyx','nova']

    voice = random.choice(voices)
    response = openai.audio.speech.create(
                model = "tts-1",
                voice = voice,
                input = message,
                response_format = 'wav'
                )
    
    response.stream_to_file(speech_file_path_object)


if __name__ == '__main__':


    # Agent - Model
    agent = ChargingAgent(
        name="Tesla Charging Example",
        example_qa=example_qa,
        json_filepath=params_filepath,
        quantize=None,
        model_name = "microsoft/Phi-3-mini-4k-instruct",
        evaluate=True
    )

    clear_param_backups(params_filepath)

    f = open(input_benchmark_filename)

    input_text_benchmark = json.load(f)

    results_all = []

    correct_count, incorrect_count, correct_count_audio, incorrect_count_audio = 0, 0, 0, 0

    for index, sample in enumerate(input_text_benchmark):
        result_this = {}
        result_this['truth_json'] = create_dict(extract_code(sample['json']))
        result_this['prompt'] = sample['prompt']
        result_this['index'] = index

        test_prompt = result_this['prompt']
        test_prompt = test_prompt.replace("Example Request: ", "")

        audio_path = 'temp.WAV'

        generate_audio_file(test_prompt, audio_path)
        try:
            result_this['pred_json'] = agent.chat(message = test_prompt)
        except:
            result_this['pred_json'] = None


        try:

            result_this['pred_json_audio'] = agent.chat(audio_path)
        except:
            result_this['pred_json_audio'] = None

        result_bool = True

        if result_this['pred_json'] is not None:
            for key in result_this['truth_json'].keys():
                if key not in result_this['pred_json'].keys() or result_this['pred_json'][key] != result_this['truth_json'][key]:
                    result_bool = False
                    break
        else:
            result_bool = False

        result_bool_audio = True
        
        if result_this['pred_json_audio'] is not None:
            for key in result_this['truth_json'].keys():
                if key not in result_this['pred_json_audio'].keys() or result_this['pred_json_audio'][key] != result_this['truth_json'][key]:
                    result_bool_audio = False
                    break
        else:
            result_bool_audio = False


        result_this['result_flag'] = result_bool
        result_this['result_flag_audio'] = result_bool_audio

        correct_count = correct_count + int(result_this['result_flag'])
        incorrect_count = incorrect_count + (1 - int(result_this['result_flag']))

        correct_count_audio = correct_count_audio + int(result_this['result_flag_audio'])
        incorrect_count_audio = incorrect_count_audio + (1 - int(result_this['result_flag_audio']))

        if not result_this['result_flag'] or not result_this['result_flag_audio']:
            if result_this['pred_json'] is not None:
                print("predicted:" )
            
                print(result_this['pred_json'])

            if result_this['pred_json_audio'] is not None:
                print("predicted from audio:" )
                print(result_this['pred_json_audio'])

            print("actual:")
            print(result_this['truth_json'])

            print("prompt: " + result_this['prompt']+"\n")

        results_all.append(result_this)

    with open(results_filename, 'w') as f:
        json.dump(results_all, f, indent = 4)

    print(f"correct_count:{str(correct_count)}, incorrect_count: {str(incorrect_count)}")
    print(f"correct_count_audio:{str(correct_count_audio)}, incorrect_count_audio: {str(incorrect_count_audio)}")


"""
Gemma 7b - 17GB, 97%, acc 100% correct
8bit - 16GB - 70%, acc 100% correct
4 bit - 8gb 50% 88/96 correct

Gemma 7b-it 4 bit 7GB 50%, acc 100% correct

Gemma 2b  6GB 50%

"""

"""
benchmark_size: 198

whisper - 900MB, 19%

4 bit - 3 GB 20% correct 192 incorrect 6


predicted:
{'end_charge_level': 0.31, 'end_charge_time': '1219'}
actual:
{'end_charge_time': '0019', 'end_charge_level': 0.31}
prompt: We need the car at 31% battery by 12:19 AM.

predicted:
{'end_charge_level': 0.41, 'end_charge_time': '1216'}
actual:
{'end_charge_time': '0016', 'end_charge_level': 0.41}
prompt: We need the car at 41% battery by 12:16 AM.

predicted:
{'end_charge_level': 0.57, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.57}
prompt: We need the car at 57% battery by 1:42 PM for work.

predicted:
{'end_charge_level': 0.47, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.47}
prompt: We need the car at 47% battery by 1:42 PM.

predicted:
{'end_charge_level': 0.58, 'end_charge_time': '1216'}
actual:
{'end_charge_time': '0016', 'end_charge_level': 0.58}
prompt: We need the car at 58% battery by 12:16 AM for the overnight charging.

{"end_charge_level": 0.88,
}
predicted:
None
actual:
{'end_charge_level': 0.88}
prompt: We need the car at 88% battery when going to work tomorrow.






8 bit - 5 GB 22% correct: 193 incorrect: 5

predicted:
{'end_charge_level': 0.31, 'end_charge_time': '1219'}
actual:
{'end_charge_time': '0019', 'end_charge_level': 0.31}
prompt: We need the car at 31% battery by 12:19 AM.

predicted:
{'end_charge_level': 0.57, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.57}
prompt: We need the car at 57% battery by 1:42 PM for work.

predicted:
{'end_charge_level': 0.47, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.47}
prompt: We need the car at 47% battery by 1:42 PM.

predicted:
{'end_charge_time': '0142'}
actual:
{'end_charge_time': '1342'}
prompt: We need the car charged by 1:42 PM for the meeting.

predicted:
{'end_charge_level': 1.0, 'end_charge_time': '2319'}
actual:
{'end_charge_time': '2019'}
prompt: We need the car fully charged for the party at 8:19 PM.



no quantization - 8 GB 45%  incorrect 4/198, correct 194/198



predicted:
{'end_charge_level': 0.31, 'end_charge_time': '1219'}
actual:
{'end_charge_time': '0019', 'end_charge_level': 0.31}
prompt: We need the car at 31% battery by 12:19 AM.

predicted:
{'end_charge_level': 0.57, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.57}
prompt: We need the car at 57% battery by 1:42 PM for work.

predicted:
{'end_charge_level': 0.47, 'end_charge_time': '1442'}
actual:
{'end_charge_time': '1342', 'end_charge_level': 0.47}
prompt: We need the car at 47% battery by 1:42 PM.

predicted:
{'end_charge_time': '0142'}
actual:
{'end_charge_time': '1342'}
prompt: We need the car charged by 1:42 PM for the meeting.
"""