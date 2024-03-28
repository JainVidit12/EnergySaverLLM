# %% Importing stuff


# import auxillary packages
import json
from typing import Optional, List, Dict


from Agent import ChargingAgent, extract_code, reset_params_file, clear_param_backups

# global params_filepath
path_prefix = '/scr1/vjain018/EnergySaverLLM/CustomModelLLM/'

params_filepath = path_prefix+"params/EVCharging.json"
params_filepath_original = path_prefix+"params/EVCharging_original.json"


# %% Few Shot Prompt
example_qa = """
----------
Instruction: Charge the car till 9 AM.
Answer Code:
```JSON
"end_charge_time": 9
```

----------
Question: Charge the car by 3 PM
Answer Code:
```JSON
"end_charge_time": 15
```
"""


if __name__ == '__main__':


    # Agent - Model
    agent = ChargingAgent(
        name="Tesla Charging Example",
        example_qa=example_qa,
        json_filepath=params_filepath,
        # evaluate=True
    )

    clear_param_backups(params_filepath)

    user_message = input("Input Command or path to audio FLAC file, enter 'quit' to cancel: ")

    while user_message != "quit":
        print(agent.chat(message=user_message))
        
        reset_params_file(params_filepath, params_filepath_original)


        user_message = input("Input Command or path to audio FLAC file, enter 'quit' to cancel: ")