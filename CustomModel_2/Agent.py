from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import Optional
import json
import datetime
import os
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np


from AudioProcessor import AudioProcessor

SYSTEM_MSG = """<|system|>
You are a chatbot to 
(1) write JSON to edit parameters as per user request for Electric Vehicle Charging.

--- JSON ---
{json} 

Only mention parameters requested by the user. 
If charge level or time is not mentioned by the user, DO NOT MENTION IT. The prevailing value will persist.
ONLY MENTION PARAMETERS TO BE CHANGED. Do not repeat the description fields. 
You just need to write JSON snippet in ```JSON ...``` block.

<|end|>

{example_qa}

<|user|>
{request}<|end|>

<|assistant|>
"""

SYSTEM_MSG_TIME_ADJUST = """<|system|>
You are a chatbot to adjust the user request text. The text might contain one or more of the following:

 - A percentage value
 - A time value

You might need to make one or more of the following adjustments:
(1) If the request mentions a time, the last two digits are ALWAYS the minute digits. Insert a colon(:) before the minute digits, or replace the dot(.) with a colon (:).
 THERE SHOULD ALWAYS BE TWO DIGITS AFTER THE COLON (:).
(2) If the request has a whitespace between two numerical digits followed by a %, remove the whitespace.
DO NOT CHANGE THE DIGIT VALUES, ONLY INSERT OR REMOVE CHARACTERS BASED ON THE ABOVE INSTRUCTIONS. The times are in 12 Hour format, DO NOT CONVERT TO 24 Hour format.
DO NOT MIX PERCENTAGE DIGITS WITH TIME DIGITS.

<|end|>

<|user|>
Charge the car till 9 0%.
<|end|>
<|assistant|>
Charge the car till 90%
<|end|>

<|user|>
Charge the car by 215 PM.
<|end|>
<|assistant|>
Charge the car by 2:15 PM.
<|end|>

<|user|>
Charge the car by 1015 AM.
<|end|>
<|assistant|>
Charge the car by 10:15 AM.
<|end|>

<|user|>
{request}<|end|>

<|assistant|>
"""
class CustomGenerationPipeline():
    def __init__(
        self,
        model,
        tokenizer,
        save_logits : Optional[bool],
        device : Optional[str],
        
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._save_logits = False
        
        if save_logits is not None:
            self._save_logits = save_logits
            if self._save_logits:
                self.df_logits = pd.DataFrame()
        self._device = device
        

    def __call__(
        self,
        message : str,
        addl_logs : Optional[dict],
        generation_config : dict = {
            "max_new_tokens": 50,
            
            "output_scores": True,
            "return_dict_in_generate" : True
        },
    ):
        input_ids = self._tokenizer(message, return_tensors="pt").to(self._device)
        prompt_length = input_ids['input_ids'].shape[1]


        outputs = self._model.generate(**input_ids, **generation_config)
        generated_seq = outputs.sequences[0, prompt_length:]

        generated_text = self._tokenizer.decode(generated_seq, skip_special_tokens=True)
        generated_text = generated_text.split('\n')[0]
        if generation_config.get('return_dict_in_generate', False) and generation_config.get('output_scores', False):
            logits = outputs.scores
            if self._save_logits:
                
                
                # Convert logits to a numpy array
                logits_array = np.array([logit.cpu().numpy() for logit in logits])

                # Reshape the logits to match the format (steps, vocabulary size)
                num_steps = logits_array.shape[0]
                vocab_size = logits_array.shape[-1]
                logits_reshaped = logits_array.reshape(num_steps, vocab_size)

                # Get the generated token IDs (excluding the initial input)
                generated_token_ids = generated_seq.cpu().numpy()

                # Calculate perplexity for each generated token
                perplexities = []
                winning_tokens = []
                for step, token_id in enumerate(generated_token_ids):
                    logit_step = logits_reshaped[step]
                    prob_step = F.softmax(torch.tensor(logit_step), dim=-1).numpy()
                    token_prob = prob_step[token_id]

                    highest_prob_token_id = prob_step.argmax()
                    highest_prob_token = self._tokenizer.decode([highest_prob_token_id], skip_special_tokens = False)
                    winning_tokens.append(highest_prob_token)

                    token_perplexity = np.exp(-np.log(token_prob))
                    perplexities.append(token_perplexity)

                # Create a DataFrame from the logits
                df_logits = pd.DataFrame(logits_reshaped)

                # Optional: Add column names corresponding to token IDs
                df_logits.columns = [f"token_{i}" for i in range(vocab_size)]

                # Add the perplexities column
                df_logits['perplexity'] = perplexities
                df_logits['winning_token'] = winning_tokens
                df_logits['final_text'] = generated_text

                for key, value in addl_logs.items():
                    df_logits[key] = value


                self.df_logits = pd.concat([self.df_logits, df_logits], ignore_index=True)
            else:
                return outputs
        
        return generated_text

    def saveLogits(
        self,
        filename : str,
        clear : bool = False
    ):
        self.df_logits.to_csv('filename')
        if clear:
            self.df_logits = self.df_logits.iloc[0:0]

    def getCompactPPL(
        self,
        addl_cols : Optional[list] = []
    ):
        addl_cols.extend(['winning_token','perplexity','final_text'])
        if self._save_logits:
            return self.df_logits[addl_cols]
        return None



class ChargingAgent():

    def __init__(
        self,
        model_name : Optional[str] = "microsoft/Phi-3-mini-4k-instruct",
        example_qa="",
        json_filepath="",
        evaluate=False,
        quantize = None,
        generation_args : Optional[dict] = {
                "max_new_tokens": 50,
                "return_full_text": False,
                "temperature": 0.0001,
                "do_sample": True,
                "output_scores": True,
            },
        get_logits : Optional[bool] = False,
    ):

        

        self._example_qa = example_qa
        self._evaluate = evaluate

        if quantize is not None:
            self._model = AutoModelForCausalLM.from_pretrained(model_name, 
                                device_map=0, 
                                quantization_config=quantize,
                                torch_dtype=torch.float16,
                                attn_implementation = "flash_attention_2",
                                trust_remote_code=True)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_name, 
                                device_map=0,
                                torch_dtype=torch.float16,
                                attn_implementation = "flash_attention_2",
                                trust_remote_code=True).to(0)
        

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._pipe = CustomGenerationPipeline(
            model = self._model,
            tokenizer = self._tokenizer,
            device = "cuda",
            save_logits=get_logits
        )

        self._generation_args = generation_args
        
        self._audio_processor = AudioProcessor(gpu_no = 1)

        self._json_filepath = json_filepath
        with open(json_filepath, 'r') as f:
            self._json_str = f.read()

    def chat_intermediate(
        self,
        message : str,
        addl_logs : Optional[dict],
    ):
        message_to_adjust = SYSTEM_MSG_TIME_ADJUST.format(
                request = message
            )

        message = self._pipe(
            message_to_adjust, 
            generation_config = self._generation_args, 
            addl_logs = addl_logs
            )

        # message = intermediate_output[0]['generated_text']
        message = message.partition('\n')[0]  
        return message

    def chat(
        self,
        message : str
    ):
        if '.wav' in message  or '.WAV' in message:
            message = self._audio_processor.processAudio(message)
        
            message = chat_intermediate(message)
        
        message = SYSTEM_MSG.format(
            json = self._json_str,
            example_qa = self._example_qa,
            request = message
        )
        # input_ids = self._tokenizer(message, return_tensors="pt").to("cuda")

        # outputs = self._model.generate(**input_ids, max_new_tokens = 50, return_dic)

        # output_text = self._tokenizer.decode(outputs[0])

        output_text = self._pipe(message, self._generation_args)

        # output_text = output[0]['generated_text']

        # print(f"output string: {output_text}")

        answer_idx = output_text.find('Answer JSON:')
        output_text = output_text[answer_idx:]
        try:
            extracted_json = extract_code(output_text)
        except:
            return None

        # print(f"processed output: {extracted_json}")

        if self._evaluate:
            return create_dict(extracted_json)
        
        self._json_str = _insert_params(self._json_str, extracted_json)

        # print(self._json_str)
        _replace_json(self._json_str, self._json_filepath)

        return output_text
    
    
    def getModel(self):
        return self._model
    def getPipe(self):
        return self._pipe


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
    if pos2 == -1:
        pos2 = input_text.find(',')
        input_text = input_text[:pos2]
    else:
        input_text = input_text[:pos2]
    
    pos3 = input_text.find('\n')
    input_text = input_text[pos3+1:]

    return input_text

def create_dict(new_param_str: str) -> dict:
    if new_param_str[0] != '{':
        new_param_str = '{' + new_param_str + '}'
    elif new_param_str[-1] != '}':
        new_param_str = new_param_str + '\n}'

    # print(new_param_str)
    
    try:
        return json.loads(new_param_str)
    except:
        print(new_param_str)

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
