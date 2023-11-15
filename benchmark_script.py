import gurobipy as gp
from gurobipy import GRB
from eventlet.timeout import Timeout

# import auxillary packages
import requests  # for loading the example source code
import openai
import json
import datetime

# import flaml and autogen
from flaml import autogen
from flaml.autogen.agentchat import Agent, UserProxyAgent
from EnergySaverLLM.Agent import ChargingAgent, reset_params_file
from flaml.autogen.code_utils import extract_code

params_filepath = "EnergySaverLLM/Model/params/EVCharging.json"
params_filepath_backup = "EnergySaverLLM/Model/params/EVCharging_original.json"

reset_params_file(params_filepath, params_filepath_backup)

config_list = autogen.config_list_from_json(
    env_or_file = "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo-16k"],
    },
)

autogen.oai.ChatCompletion.start_logging()

code_path = "EnergySaverLLM/Model/EVCharging.py"

with open(code_path) as f:
    code = f.read()

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
datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

benchmark_filepath = "benchmark/EV.benchmark.json"
benchmark_ques = []
with open(benchmark_filepath, 'r') as f:
    benchmark_dict = json.loads(f.read())
    benchmark_ques = benchmark_dict['questions']


results_filepath = "benchmark/results/" + datetime_str + ".csv"

for ques in benchmark_ques:
    reset_params_file(params_filepath, params_filepath_backup)
    agent = ChargingAgent(
        name="Tesla Charging Example",
        source_code=code,
        example_qa=example_qa,
        json_filepath=params_filepath,
        bench_results_filepath = results_filepath,
        que_hashcode = ques['hash'],
        llm_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
        }
    )

    user = UserProxyAgent(
        "user", max_consecutive_auto_reply=0,
        human_input_mode="NEVER", code_execution_config=False
    )

    user.initiate_chat(agent, message=ques['QUESTION'])



