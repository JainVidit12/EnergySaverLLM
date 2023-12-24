from datetime import datetime
from gurobipy import GRB, Model, LinExpr
import json

# Fetching data from JSON file

params_filepath = "/Users/viditjain/Desktop/LLM Optimizer/EnergySaverLLM/IndustrialChargerLLM/Model/params/EVCharging.json"

stored_params = json.load(open(params_filepath))

car_ids = list(stored_params['car_params'].keys())[2:]

#%% Initializing car params
required_charge_arr = {}
max_charge_arr = {}
max_power_arr = {}
end_time_arr = {}
start_time = 12

for id in car_ids:
    car_params = stored_params['car_params'][id]
    required_charge_arr[id] = car_params['required_charge']
    max_charge_arr[id] = car_params['max_charge']
    max_power_arr[id] = car_params['max_power']
    end_time_arr[id] = car_params['end_time']

#%% Intitializing lot params
lot_params = stored_params['lot_params']

elec_costs = [(x , y) for x, y in lot_params['elec_cost'].items()]
elec_costs = dict(elec_costs)
carbon_cost = lot_params['carbon_cost']

elec_costs_next_day = elec_costs.copy()
elec_costs_next_day = [(str(int(x)+24) , y) for x, y in elec_costs_next_day.items()]
elec_costs.update(elec_costs_next_day)

carbon_cost_weight = lot_params['carbon_cost_weight']
# print(carbon_cost_weight)

elec_cost_weight = lot_params['elec_cost_weight']
# print(elec_cost_weight)

max_power = lot_params['max_power']


#%% Setting up the model

keys = elec_costs.keys()

model = Model("EVCharging")

vars = {}
obj_expr = LinExpr()

## Initializing dictionary, empty expression for each key
total_power_constr = {k : LinExpr() for k in keys}

for id in car_ids:

    end_time_H = end_time_arr[id]
    if end_time_H <= start_time:
        end_time_H += 24

    max_power_KW =  max_power_arr[id]
    required_charge_KWH = required_charge_arr[id]
    max_charge_KWH = max_charge_arr[id]

    upper_bounds = [0]*len(elec_costs)
    upper_bounds[start_time:end_time_H] = [max_power_KW] * (end_time_H - start_time)

    # Create variables
    vars[id] = model.addVars(keys,
                    vtype = GRB.INTEGER,
                    ub = upper_bounds,
                    name = id)

    # Adjust objective expression for this car
    # Elec cost, carbon cost, early charging reward
    obj_expr += (elec_cost_weight * sum(vars[id][i] * elec_costs[i] for i in keys) +
                carbon_cost_weight * sum(vars[id][i] * carbon_cost for i in keys) +
                sum(vars[id][i]*(int(i) - end_time_H) for i in keys))


    for key in keys:
        total_power_constr[key] += vars[id][key]

    # Min charge constraint, car should be charged at least upto the required charge
    model.addConstr(sum(vars[id][i] for i in keys) >= required_charge_KWH)

    # Max charge constraint, car can't be overcharged
    model.addConstr(sum(vars[id][i] for i in keys) <= max_charge_KWH)

# Total charge capacity constraint - Total power supplied to charging lot
for k in keys:
    model.addConstr(total_power_constr[k] <= max_power)

# Set objective
model.setObjective(
    obj_expr, GRB.MINIMIZE)

# Optimize model
m = model
m.optimize()

if m.status == GRB.OPTIMAL:
    all_vars = m.getVars()
    values = m.getAttr("X", all_vars)
    names = m.getAttr("VarName", all_vars)

    # for name, val in zip(names, values):
    #     print(f"Scheduled consumption at Hour {name} : {int(val):d} KWH")

else:
    print("Not solved to optimality. Optimization status:", m.status)
