from Agent import ChargingAgent
from transformers import BitsAndBytesConfig
import torch
import json
import os


# global params_filepath
path_prefix = os.path.abspath(os.getcwd())

params_filepath = path_prefix+"/params/EVCharging.json"
params_filepath_original = path_prefix+"/params/EVCharging_original.json"

fourBit_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
eightBit_config = BitsAndBytesConfig(load_in_8bit=True)

benchmark_filepath = path_prefix + '/evaluate/results_8bit.json'

addl_logging_cols = ['prompt', 'result_flag', 'result_flag_audio']

if __name__ == "__main__":

    # Agent - Model
    agent = ChargingAgent(
        example_qa="",
        json_filepath=params_filepath,
        quantize=eightBit_config,
        model_name = "microsoft/Phi-3-mini-4k-instruct",

        generation_args={
                "max_new_tokens": 50,
                "output_scores": True,
                "return_dict_in_generate" : True
            },
        get_logits=True
    )
    with open(benchmark_filepath, 'r') as f:
        benchmark_results = json.loads(f.read())
    index = 0
    with open('8Bitrun.txt') as transcript:
        for line in transcript:
            if 'transcription' in line:
                benchmark_result_this = benchmark_results[index]

                line_trimmed = line.replace('transcription:', "").strip()
                
                agent.chat_intermediate(line_trimmed, addl_logs={
                    key:benchmark_result_this[key] for key in addl_logging_cols
                })
                index = index + 1
            # break

df_logits = agent.getPipe().getCompactPPL(addl_logging_cols)
df_logits.to_csv('logits_8bit_transcriptions.csv', index=False)