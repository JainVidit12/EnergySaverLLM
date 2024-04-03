import gurobipy as gp
from gurobipy import GRB
import numpy as np
from eventlet.timeout import Timeout
import random
import time

# import auxillary packages
import requests  # for loading the example source code
import json
import openai

example_qa = """
----------
Param Update: ```JSON
"end_charge_time": 9```
Example User request: Charge the car till 9 AM.
----------
Param Update: ```JSON
"end_charge_time": 13```
Example Request: Charge the car by 1 PM.
"""

SYSTEM_MSG = """You are a chatbot to 
Given the parameters to manage charging for an EV, describe an example request which would require a given change in parameters.
Answer concisely in only one line, as if you are the user. NEVER use the variable name directly in the description.
ALWAYS include the new parameter value in the description. Try to make the example request less concise.
--- JSON ---
{json} 

Here are some example codes and their possible request examples:
--- EXAMPLES ---
{example_qa}
---

The JSON for updated parameter:
"""
user_message = """
{json}

Describe the change:
"""
param_change_json = """```JSON
"end_charge_time": {new_val}
```"""
params_filepath = "/scr1/vjain018/EnergySaverLLM/CustomModelLLM/params/EVCharging.json"


with open(params_filepath) as f:
    params_json = json.load(f)
f.close()


system_message = SYSTEM_MSG.format(example_qa = example_qa, json = params_json)

values = np.arange(0, 24, 1)

benchmark_dataset = []

client = openai.OpenAI()

for val in values:
    for _ in range(4):
        param_change_json_this = param_change_json.format(new_val = str(val))
        message = [
            {"role" : "system", "content" : system_message},
            {"role" : "user", "content" : user_message.format(json = param_change_json_this)}
        ]
        response = client.chat.completions.create( model="gpt-4", messages=message)

        prompt = response.choices[0].message.content
        # print(prompt)
        benchmark_dataset.append({
            "json":param_change_json_this,
            "prompt":prompt
        })

with open('benchmark.json', 'w') as f:
    json.dump(benchmark_dataset, f, indent=4)

