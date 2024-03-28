import re
from typing import Dict, List, Optional, Union

import json
import datetime
import os

from eventlet.timeout import Timeout
from gurobipy import GRB
from termcolor import colored
import csv

from autogen.agentchat import AssistantAgent
from autogen.agentchat.agent import Agent
from autogen.code_utils import extract_code

# %% System Messages

WRITER_SYSTEM_MSG = """You are a chatbot to 
(1) write JSON to edit parameters as per user request for Electric Vehicle Charging.

--- JSON ---
{json} 

Here are some example questions and their answers and codes:
--- EXAMPLES ---
{example_qa}
---

Note that only parameters you mention will be changed.
You just need to write JSON snippet in ```JSON ...``` block.
"""

INTERPRETER_SYSTEM_MSG = """You are a chatbot to:
(1) explain solutions from a Gurobi/Python solver.

The execution result of the original source code is below.
--- Original Result ---
{execution_result}

"""
# %%
class ChargingAgent(AssistantAgent):
    
    def __init__(self,
                 name,
                 source_code,
                 example_qa="",
                 json_filepath="",
                 evaluate=False,
                 **kwargs):
        """
        Args:
            name (str): agent name.
            source_code (str): The original source code to run.
            example_qa (str): training examples for in-context learning.
            json (str): The original JSON parameters file.
            **kwargs (dict): Please refer to other kwargs in
                [AssistantAgent](assistant_agent#__init__) and
                [ResponsiveAgent](responsive_agent#__init__).
        """
        
        super().__init__(name, **kwargs)
        self._source_code = source_code
        self._example_qa = example_qa
        self._origin_execution_result = _run_with_exec(source_code)
        self._json_filepath = json_filepath
        self._evaluate = evaluate
        
        with open(json_filepath, 'r') as f:
            self._json_str = f.read()

        self._writer = AssistantAgent("writer", llm_config=self.llm_config)
        self._interpreter = AssistantAgent("interpreter",
                                         llm_config=self.llm_config)
        

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        default_reply: Optional[Union[str, Dict]] = "",
        sender: Optional[Agent] = None,
    ) -> Union[str, Dict, None]:
        # Remove unused variables:
        # The message is already stored in self._oai_messages
        del messages, default_reply
        """Reply based on the conversation history."""
        if sender not in [self._writer, self._interpreter]:
            # Step 1: receive the message from the user
            user_chat_history = ("\nHere are the history of discussions:\n"
                                 f"{self._oai_messages[sender]}")
            writer_sys_msg = (WRITER_SYSTEM_MSG.format(
                example_qa=self._example_qa,
                json=self._json_str,
            ) + user_chat_history)
            interpreter_sys_msg = INTERPRETER_SYSTEM_MSG.format(
                execution_result=self._origin_execution_result) + user_chat_history
            self._writer.update_system_message(writer_sys_msg)
            self._interpreter.update_system_message(interpreter_sys_msg)
            self._writer.reset()
            self._interpreter.reset()
            
           
            self.initiate_chat(self._writer, message=CODE_PROMPT)
            
            _, new_params = extract_code(self.last_message(self._writer)["content"])[0]
            
            if self._evaluate:
                return ""

            self._json_str = _insert_params(self._json_str, new_params)
            _replace_json(self._json_str, self._json_filepath)
            execution_rst = _run_with_exec(self._source_code)
            
            
            print(colored(str(execution_rst), "yellow"))

            
            if type(execution_rst) in [str, int, float]:
                self.initiate_chat(self._interpreter, 
                                    message=INTERPRETER_PROMPT.format(execution_rst=execution_rst))
                return self.last_message(self._interpreter)["content"]
            else:
                reply = "Sorry I cannot interpret this result."
            
            return reply
            
# %% Helper functions to edit and run code.
# Here, we use a simplified approach to run the code snippet, which would
# replace substrings in the source code to get an updated version of code.
# Then, we use exec to run the code snippet.
# This approach replicate the evaluation section of the OptiGuide paper.


def _run_with_exec(src_code: str) -> Union[str, Exception]:
    """Run the code snippet with exec.

    Args:
        src_code (str): The source code to run.

    Returns:
        object: The result of the code snippet.
            If the code succeed, returns the objective value (float or string).
            else, return the error (exception)
    """
    locals_dict = {}
    locals_dict.update(globals())
    locals_dict.update(locals())
    
    timeout = Timeout(
        60,
        TimeoutError("This is a timeout exception, in case "
                     "GPT's code falls into infinite loop."))
    try:
        exec(src_code, locals_dict, locals_dict)
    except Exception as e:
        return e
    finally:
        timeout.cancel()

    try:
        status = locals_dict["m"].Status
        if status != GRB.OPTIMAL:
            if status == GRB.UNBOUNDED:
                ans = "unbounded"
            elif status == GRB.INF_OR_UNBD:
                ans = "inf_or_unbound"
            elif status == GRB.INFEASIBLE:
                ans = "infeasible"
                m = locals_dict["m"]
                m.computeIIS()
                constrs = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
                ans += "\nConflicting Constraints:\n" + str(constrs)
            else:
                ans = "Model Status:" + str(status)
        else:
            ans = "Optimization problem solved. The objective value is: " + str(
                locals_dict["m"].objVal)
    except Exception as e:
        return e

    return ans


def saveResult(filepath : str, hashcode : str, val : int):
    
    if not os.path.isfile(filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['HashCode','Output'])
    
    with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([hashcode, val])

def _replace_json(json_str: str, json_filepath: str):
    [json_file_loc, prev_json_filename] = json_filepath.rsplit('/', 1)
    new_json_filename = prev_json_filename.replace('.json', '_backup_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.json')

    new_json_filepath = json_file_loc + '/' + new_json_filename
    os.rename(json_filepath, new_json_filepath)
    with open(json_filepath, "w") as text_file:
        text_file.write(json_str)
    text_file.close()



def _insert_params(src_json_str : str, new_params : str) -> str:
    """change JSON params.


    Args:
        src_json_str (str): the original json
        new_params (str): params to be changed.

    Returns:
        json: the full json after replacement.
    """

    json_dict = json.loads(src_json_str)
    
    if new_params[0] != '{':
        new_params = '{' + new_params + '}'

    changed_json_dict = json.loads(new_params)
    for key, value in changed_json_dict.items():
        json_dict[key] = value
        
    return json.dumps(json_dict, indent = 4)


def reset_params_file(curr_path : str, backup_path : str):
    """change JSON params.
    Args:
        curr_path (str): the file used in code
        backup_path (str): backup file

    Returns:
        json: the full json after replacement.
    """
    os.remove(curr_path)

    with open(backup_path, 'r') as f:
            contents = f.read()

    with open(curr_path, "w") as text_file:
        text_file.write(contents)

def clear_param_backups(curr_path : str):
    """change JSON params.
    Args:
        curr_path (str): the file used in code

    Returns:
        json: the full json after replacement.
    """
    path = curr_path[:curr_path.rindex('/')]+'/'
    for filename in os.listdir(path):
        if 'backup' in filename:
            os.remove(path+filename)


CODE_PROMPT = """
Answer JSON:
"""

INTERPRETER_PROMPT = """Here are the execution results: {execution_rst}

Can you organize these information to a human readable answer?
Remember to compare the new results to the original results you obtained in the
beginning. A lower objective value is better here.

--- HUMAN READABLE ANSWER ---
"""
