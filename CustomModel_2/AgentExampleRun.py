# %% Importing stuff
import torch

# import auxillary packages
import json
from typing import Optional, List, Dict


from Agent import ChargingAgent, extract_code, reset_params_file, clear_param_backups
from transformers import BitsAndBytesConfig

# global params_filepath
path_prefix = '/scr1/vjain018/EnergySaverLLM/CustomModel_2/'

params_filepath = path_prefix+"params/EVCharging.json"
params_filepath_original = path_prefix+"params/EVCharging_original.json"


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
Instruction: Charge the car till 0215 AM.
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_time": "0215"
```
<|end|>

<|user|>
Question: Charge the car till 80% by 0345 PM
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_level": 0.8,
"end_charge_time": "1545"
```
<|end|>

<|user|>
Question: Charge the car till 89% by 1045 AM
<|end|>
<|assistant|>
Answer JSON:
```JSON
"end_charge_level": 0.89,
"end_charge_time": "1045"
```
<|end|>

"""

fourBit_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
eightBit_config = BitsAndBytesConfig(load_in_8bit=True)

if __name__ == '__main__':


    # Agent - Model
    agent = ChargingAgent(
        name="Tesla Charging Example",
        example_qa=example_qa,
        json_filepath=params_filepath,
        quantize=fourBit_config,
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        # evaluate=True
    )

    clear_param_backups(params_filepath)

    user_message = input("Input Command or path to audio WAV file, enter 'quit' to cancel: ")

    while user_message != "quit":
        print(agent.chat(message=user_message))
        
        reset_params_file(params_filepath, params_filepath_original)


        user_message = input("Input Command or path to audio FLAC file, enter 'quit' to cancel: ")