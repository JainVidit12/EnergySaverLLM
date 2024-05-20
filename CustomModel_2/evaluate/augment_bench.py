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
Original Prompt: Charge the car by 8:24 AM
Changed prompt: Charge the car by 8:24 in the morning.
----------
Original Prompt: Charge the car by 7:50 PM
Changed prompt: Charge the car by evening 7:50.
----------
Original Prompt: Charge the car by 12:00 AM
Changed prompt: Charge the car by midnight.
"""

SYSTEM_MSG = """You are a chatbot to 
Given a query with a time component, rephrase the query making the following change:
 - Replace the 'AM/PM' with 'morning/evening'
 - Replace 12 AM with midnight and 12 PM with noon
--- EXAMPLES ---
{example_qa}
---
"""
user_message = """
Original Prompt:
{prompt}

Adjusted prompt:
"""


system_message = SYSTEM_MSG.format(example_qa = example_qa)

client = openai.OpenAI()

model = "gpt-4"


class Augment():
    def __init__(
        self,
        client = openai.OpenAI(),
        model : str = 'gpt-4',
        system_message : str = SYSTEM_MSG,
        example_qa : str = example_qa,
        message_template : str = user_message
    ):
        self._client = client
        self._model = model
        self._system_message = system_message.format(example_qa = example_qa)
        self._message_template = message_template


    def perform_augment(self, prompt):
        adj_prompt = self._message_template.format(prompt = prompt)

    
        
        message = [
            {"role" : "system", "content" : self._system_message},
            {"role" : "user", "content" : adj_prompt}
        ]
        response = self._client.chat.completions.create(model = self._model, messages=message)


        return response.choices[0].message.content


