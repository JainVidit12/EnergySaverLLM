# %% Importing stuff


# import auxillary packages
import json
from typing import Optional, List, Dict


from Agent import ChargingAgent, extract_code, reset_params_file, clear_param_backups, create_dict

# global params_filepath
params_filepath = "params/EVCharging.json"
params_filepath_original = "params/EVCharging_original.json"

path_prefix = '/scr1/vjain018/EnergySaverLLM/CustomModelLLM/evaluate/'
input_benchmark_filename = path_prefix + 'EV_combined.benchmark.json'
results_filename = path_prefix + 'results.json'


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
        evaluate=True
    )

    clear_param_backups(params_filepath)

    f = open(input_benchmark_filename)

    input_text_benchmark = json.load(f)

    results_all = []

    correct_count, incorrect_count = 0, 0

    for sample in input_text_benchmark:
        if 'end_charge_time' not in sample['json_str'] or sample['index']>=118:
            continue
        
        result_this = {}
        result_this['truth_json'] = create_dict(extract_code(sample['json_str']))
        result_this['prompt'] = sample['prompt']
        result_this['index'] = sample['index']

        result_this['pred_json'] = agent.chat(message = result_this['prompt'])

        result_this['result_flag'] = result_this['truth_json'] == result_this['pred_json']

        correct_count = correct_count + int(result_this['result_flag'])
        incorrect_count = incorrect_count + (1 - int(result_this['result_flag']))

        results_all.append(result_this)

    with open(results_filename, 'w') as f:
        json.dump(results_all, f, indent = 4)

    print(f"correct_count:{str(correct_count)}, incorrect_count: {str(incorrect_count)}")
