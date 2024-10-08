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

import pandas as pd
from AudioProcessor import AudioProcessor
from augment_bench import Augment


example_qa = """
----------
Param Update: ```JSON
"end_charge_time": "0900"```
Example Request: We will leave for the event at 9:00 AM.
----------
Param Update: ```JSON
"end_charge_time": "1345",
"end_charge_level":0.89```
Example Request: We need the car at 89% battery by 1:45 PM.
----------
Param Update: ```JSON
"end_charge_time": "1945",
"end_charge_level":0.89```
Example Request: We need the car at 89% battery by 7:45 PM.
----------
Param Update: ```JSON
"end_charge_level":1```
Example Request: We need the car at full battery for our 2000 mile trip.
"""

SYSTEM_MSG = """You are a chatbot to 
Given the parameters to manage charging for an EV, describe an example request which would require a given change in parameters.
Answer concisely in only one line, as if you are the user. NEVER use the variable name directly in the description.
ALWAYS include the new parameter value in the description. You can mention the an end location for example school, work, event, party.
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

Example Request:
"""
param_change_json_time = """```JSON
"end_charge_time": "{new_val_time}"
```"""
param_change_json_level = """```JSON
"end_charge_level": {new_val_level}
```"""
param_change_json_both = """```JSON
"end_charge_time": {new_val_time},
"end_charge_level": {new_val_level}
```"""

params_filepath = "/scr1/vjain018/EnergySaverLLM/CustomModel_2/params/EVCharging.json"


with open(params_filepath) as f:
    params_json = json.load(f)
f.close()


system_message = SYSTEM_MSG.format(example_qa = example_qa, json = params_json)

values_time = np.arange(0, 24, 1)


benchmark_dataset = pd.DataFrame(columns = ['prompt','json','transcribed_prompt', 'hour_val', 'level_val', 'is_adjusted'])

client = openai.OpenAI()

model = "gpt-4"

audio_processor = AudioProcessor()
augmentation = Augment(client = client, model = model)

for hour_val in values_time:
    minutes_sampled = np.random.randint(0, 59, 10)
    for minute in minutes_sampled:
        values_level = np.random.randint(10, 100, 10)
        for level_val in values_level:
            
            minute_str = str(minute)
            if minute < 10:
                minute_str = "0" + minute_str
            
            hour_str = str(hour_val)
            if hour_val < 10:
                hour_str = "0" + hour_str

            timestamp = hour_str + minute_str


            val_str = "{:.2f}".format(level_val / 100.0)
            
            param_change_json = param_change_json_both.format(new_val_time = timestamp, new_val_level = val_str)
            
            message = [
                {"role" : "system", "content" : system_message},
                {"role" : "user", "content" : user_message.format(json = param_change_json)}
            ]
            response = client.chat.completions.create( model=model, messages=message)

            prompt = response.choices[0].message.content

            transcribed_prompt = audio_processor.generate_transcribed_text(prompt)

            adjusted_prompt = augmentation.perform_augment(prompt)

            transcribed_adjusted_prompt = audio_processor.generate_transcribed_text(adjusted_prompt)

            benchmark_dataset.loc[len(benchmark_dataset)] = {
                    'prompt' : prompt,
                    'json' : param_change_json,
                    'transcribed_prompt': transcribed_prompt,
                    'hour_val' : hour_val,
                    'level_val' : level_val,
                    'is_adjusted' : False
                }
            
            benchmark_dataset.loc[len(benchmark_dataset)] = {
                    'prompt' : adjusted_prompt,
                    'json' : param_change_json_both,
                    'transcribed_prompt': transcribed_adjusted_prompt,
                    'hour_val' : hour_val,
                    'level_val' : level_val,
                    'is_adjusted' : True
                }
            # print(benchmark_dataset.head())
            if len(benchmark_dataset) % 100 == 0:
                print(len(benchmark_dataset))
    #         break
    #     break
    # break
    benchmark_dataset.to_parquet('benchmark_large.parquet')
    with open('benchmark_large.json', 'w') as f:
        json.dump(benchmark_dataset.to_dict('records'), f, indent=4)
    print("written")





# for val in values_level:
    
#     val_str = "{:.2f}".format(val / 100.0)
#     param_change_json_this = param_change_json_level.format(new_val_level = val_str)
    
#     message = [
#         {"role" : "system", "content" : system_message},
#         {"role" : "user", "content" : user_message.format(json = param_change_json_this)}
#     ]
#     response = client.chat.completions.create( model=model, messages=message)

#     prompt = response.choices[0].message.content
#     benchmark_dataset_level.append({
#         "json":param_change_json_this,
#         "prompt":prompt
#     })
#     # break
    
# for _ in range(100):
#     sample_time = random.choice(benchmark_dataset_time)['json']
#     sample_level = random.choice(benchmark_dataset_level)['json']

#     new_json = sample_time + sample_level

#     new_json = new_json.replace("\n``````JSON", ",")

#     # print(sample)

#     message = [
#         {"role" : "system", "content" : system_message},
#         {"role" : "user", "content" : user_message.format(json = new_json)}
#     ]
#     response = client.chat.completions.create( model=model, messages=message)

#     prompt = response.choices[0].message.content
#     benchmark_dataset.append({
#         "json":new_json,
#         "prompt":prompt
#     })
#     # break
    
# benchmark_dataset.extend(benchmark_dataset_level)
# benchmark_dataset.extend(benchmark_dataset_time)

# with open('benchmark_large.json', 'w') as f:
#     json.dump(benchmark_dataset, f, indent=4)

