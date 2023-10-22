from datetime import datetime
from gurobipy import GRB, Model
import json
from utils import getCurrentBattery


# Fetching data from JSON file
stored_params = json.load(open('params/EVCharging.json'))

start_charge_level = getCurrentBattery()
end_charge_level = stored_params['default_end_charge']

total_battery_capacity = stored_params['battery_capacity']*1000

this_charge_WH = (end_charge_level-start_charge_level)*total_battery_capacity

max_power_W = stored_params['max_power']*1000

curr_time_H = datetime.now().hour + 1
end_time_H = stored_params["end_charge_time"]

elec_costs = [(x , y / 1000) for x, y in stored_params['elec_cost'].items()]
elec_costs = dict(elec_costs)

carbon_costs = [(x , y / 1000) for x, y in stored_params['carbon_cost'].items()]
carbon_costs = dict(carbon_costs)

print(elec_costs)

# If end_time is in the next day, append the cost array to itself
# In the new array, index 25 means 1 AM in the next day.
if(end_time_H <= curr_time_H):
    end_time_H += 24

    elec_costs_next = elec_costs.copy()
    elec_costs_next = [(str(int(x)+24) , y) for x, y in elec_costs_next.items()]
    elec_costs.update(elec_costs_next)

    carbon_costs_next = carbon_costs.copy()
    carbon_costs_next = [(str(int(x)+24) , y) for x, y in carbon_costs_next.items()]
    carbon_costs.update(carbon_costs_next)

elec_costs = elec_costs

max_elec_cost = stored_params['max_energy_cost']
carbon_cost_weight = stored_params['carbon_cost_weight']

upper_bounds = [0]*len(elec_costs)
upper_bounds[curr_time_H:end_time_H] = [max_power_W] * (end_time_H - curr_time_H)

keys = elec_costs.keys()

# Create a new model
model = Model("EVCharging")

# OPTIGUIDE DATA CODE GOES HERE

print(str(len(upper_bounds))+"\n")
# Create variables
x = model.addVars(keys,
                  vtype = GRB.INTEGER,
                  ub = upper_bounds,
                  name="x")

# Set objective
model.setObjective(
    sum(x[i] * elec_costs[i] for i in keys) +
    carbon_cost_weight * sum(x[i] * carbon_costs[i] for i in keys), GRB.MINIMIZE)

# Total cost constraint
model.addConstr(sum(x[i] * elec_costs[i] for i in keys) <= max_elec_cost)

# Total charge constraint
model.addConstr(sum(x[i] for i in keys) == this_charge_WH)


# Optimize model
model.optimize()
for v in model.getVars():
    print(f"{v.varName} = {v.x}\n")

# OPTIGUIDE CONSTRAINT CODE GOES HERE

Solve
m.update()
model.optimize()

print(time.ctime())
if m.status == GRB.OPTIMAL:
    print(f'Optimal cost: {m.objVal}')
else:
    print("Not solved to optimality. Optimization status:", m.status)
