import gurobipy as gp
from gurobipy import GRB
import numpy as np
from eventlet.timeout import Timeout
import random
import time

# import auxillary packages
import requests  # for loading the example source code
import openai
import json

# import flaml and autogen
from flaml import autogen
from flaml.autogen.agentchat import Agent, UserProxyAgent
from flaml.autogen.code_utils import extract_code
from create_bench_agent import CreateBenchAgent


example_qa = """
----------
Param Update: ```JSON
"end_charge_time": 9```
Example User request: Charge the car till 9 AM.
----------
Param Update: ```JSON
"end_charge": 1.00```
Example Request: Charge the car to full charge.
----------
Param Update: ```JSON
"end_charge": 1.00```
Example Request: Charge the car to full charge.
"""

