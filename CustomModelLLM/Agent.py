from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
import json
import datetime
import os

SYSTEM_MSG = """You are a chatbot to 
(1) write JSON to edit parameters as per user request for Electric Vehicle Charging.

--- JSON ---
{json} 

Here are some example questions and their answers and codes:
--- EXAMPLES ---
{example_qa}
---

Note that only parameters you mention will be changed.
You just need to write JSON snippet in ```JSON ...``` block.

Current time: 11 AM

Here is the user request:
{request}

Answer JSON:
"""

class ChargingAgent():

    def __init__(
        self,
        model_name : Optional[str] = "google/gemma-7b",
        example_qa="",
        json_filepath="",
        evaluate=False,
        quantize = None,
        **kwargs
    ):

        

        self._example_qa = example_qa
        self._evaluate = evaluate

        if quantize is not None:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        

        self._json_filepath = json_filepath
        with open(json_filepath, 'r') as f:
            self._json_str = f.read()

    def chat(
        self,
        message : str
    ):
        message = SYSTEM_MSG.format(
            json = self._json_str,
            example_qa = self._example_qa,
            request = message
        )
        input_ids = self._tokenizer(message, return_tensors="pt").to("cuda")

        outputs = self._model.generate(**input_ids, max_new_tokens = 20)

        output_text = self._tokenizer.decode(outputs[0])

        answer_idx = output_text.find('Answer JSON:')
        output_text = output_text[answer_idx:]

        extracted_json = extract_code(output_text)

        if self._evaluate:
            return create_dict(extracted_json)
        
        self._json_str = _insert_params(self._json_str, extracted_json)

        # print(self._json_str)
        _replace_json(self._json_str, self._json_filepath)

        return output_text


def _replace_json(json_str: str, json_filepath: str):
    [json_file_loc, prev_json_filename] = json_filepath.rsplit('/', 1)
    new_json_filename = prev_json_filename.replace('.json', '_backup_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.json')

    new_json_filepath = json_file_loc + '/' + new_json_filename
    os.rename(json_filepath, new_json_filepath)
    
    
    with open(json_filepath, "w") as text_file:
        json_dict = json.loads(json_str)
        json.dump(json_dict, text_file, indent=4)
    text_file.close()
    


def extract_code(input_text: str) -> str:
    pos1 = input_text.find('```')
    input_text = input_text[pos1+3:]

    pos2 = input_text.find('```')
    input_text = input_text[:pos2]
    
    pos3 = input_text.find('\n')
    input_text = input_text[pos3+1:]

    return input_text

def create_dict(new_param_str: str) -> dict:
    if new_param_str[0] != '{':
        new_param_str = '{' + new_param_str + '}'

    return json.loads(new_param_str)

def _insert_params(src_json_str : str, new_params : str) -> str:
    """change JSON params.


    Args:
        src_json_str (str): the original json
        new_params (str): params to be changed.

    Returns:
        json: the full json after replacement.
    """

    json_dict = json.loads(src_json_str)
    
    changed_json_dict = create_dict(new_params)
    
    for key, value in changed_json_dict.items():
        json_dict[key] = value
    
    # print(json_dict)

    return json.dumps(json_dict, indent = 4)


def reset_params_file(curr_path : str, backup_path : str):
    """Reset parameters file to the original params
    """
    os.remove(curr_path)

    with open(backup_path, 'r') as f:
            contents = f.read()

    with open(curr_path, "w") as text_file:
        text_file.write(contents)

def clear_param_backups(curr_path : str):
    """Change previous param change records
    """
    path = curr_path[:curr_path.rindex('/')]+'/'
    for filename in os.listdir(path):
        if 'backup' in filename:
            os.remove(path+filename)
