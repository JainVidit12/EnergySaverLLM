import re
from typing import Dict, List, Optional, Union

import json
import datetime
import os

from eventlet.timeout import Timeout
from gurobipy import GRB
from termcolor import colored
import csv

from flaml.autogen.agentchat import AssistantAgent
from flaml.autogen.agentchat.agent import Agent
from flaml.autogen.code_utils import extract_code

# %% System Messages

SYSTEM_MSG = """You are a chatbot to 
Given the parameter descriptions to manage EV charging lot, describe an example request which would require a given change in parameters.
Answer concisely in only one line, as if you are the user. NEVER use the variable name directly in the description.
ALWAYS include the new parameter value in the description.
--- JSON ---
{json} 

Here are some example codes and their possible request examples:
--- EXAMPLES ---
{example_qa}
---

The JSON for updated parameter:

"""
# %%
class CreateBenchAgent(AssistantAgent):
    
    def __init__(self,
                 name,
                 example_qa="",
                 json_str="",
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
        self._example_qa = example_qa
        self._json_str = json_str
        

        self._writer = AssistantAgent("writer", llm_config=self.llm_config)

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
        if sender not in [self._writer]:
            # Step 1: receive the message from the user
            user_chat_history = ("\nHere are the history of discussions:\n"
                                 f"{self._oai_messages[sender]}")
            writer_sys_msg = (SYSTEM_MSG.format(
                example_qa=self._example_qa,
                json=self._json_str,
            ) + user_chat_history)
            
            self._writer.update_system_message(writer_sys_msg)
            self._writer.reset()
           
            self.initiate_chat(self._writer, message=CODE_PROMPT)
            return self.last_message(self._writer)["content"]

# %%
CODE_PROMPT = """


Describe the change:

"""
    