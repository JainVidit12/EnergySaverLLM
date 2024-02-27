# %% Importing stuff

import gurobipy as gp
from gurobipy import GRB
from eventlet.timeout import Timeout

# import auxillary packages
import openai
import json
from typing import Optional, List, Dict

# import autogen
from autogen.agentchat import Agent, UserProxyAgent
from autogen.code_utils import extract_code
from autogen import config_list_from_json, runtime_logging

from Agent import ChargingAgent, reset_params_file
from Model.utils import getInput

# global params_filepath
params_filepath = "Model/params/EVCharging.json"
params_filepath_backup = "Model/params/EVCharging_original.json"

code_path = "Model/EVCharging.py"
with open(code_path) as f:
    code = f.read()

# %% Logging
log_history = {}
runtime_logging.start()

# %% Config Lists, use corresponding for different model
config_list_gpt3_5 = config_list_from_json(
    env_or_file = 'CONFIG_JSON',
    filter_dict={
        "model": ["gpt-3.5-turbo-16k"],
    },
)

config_list_gpt4 = config_list_from_json(
    env_or_file = 'CONFIG_JSON',
    filter_dict={
        "model": ["gpt-4"],
    },
)

# %% Few Shot Prompt
example_qa = """
----------
Instruction: Charge the car till 9 AM.
Answer Code:
```JSON
"end_charge_time": 9
```

----------
Question: Charge the car to full charge by 9 AM
Answer Code:
```JSON
"end_charge": 1.00,
"end_charge_time": 9
```
"""

def terminate_conversation(msg : Optional[List[Dict]]):
    return True


if __name__ == '__main__':

    # Reset params file to the original file
    reset_params_file(params_filepath, params_filepath_backup)

    # Agent - Model
    agent = ChargingAgent(
        name="Tesla Charging Example",
        source_code=code,
        example_qa=example_qa,
        json_filepath=params_filepath,
        llm_config={
            # "request_timeout": 600,
            "seed": 42,
            "config_list": config_list_gpt4,
        }
    )

    # User - takes human input
    user = UserProxyAgent(
        "user",
        is_termination_msg=terminate_conversation,
        human_input_mode="NEVER", code_execution_config=False
    )

    user_message = getInput(input("Input Command or path to audio FLAC file, enter 'quit' to cancel: "))

    while user_message != "quit":
        user.initiate_chat(agent, message=user_message)
        
        # Reset params file to the original file
        reset_params_file(params_filepath, params_filepath_backup)
        
        user_message = input("Input Command or path to audio FLAC file, enter 'quit' to cancel: ")

    
    
