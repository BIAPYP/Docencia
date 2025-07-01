from pulp import *
import numpy as np

# Initialize the problem
prob = LpProblem("Refinery_Optimization", LpMaximize)

# Define sets and parameters
input_streams = ['crude_A', 'crude_B', 'crude_C']
blending_units = ['blender_1', 'blender_2']
products = ['gasoline', 'diesel', 'fuel_oil']
temp_warehouses = ['warehouse_1', 'warehouse_2']
selling_points = ['market_1', 'market_2', 'market_3']

# Input streams properties and costs ($/barrel)
input_costs = {
    'crude_A': 65,
    'crude_B': 60,
    'crude_C': 70
}

# Define total availability for each crude
crude_availability = {
    'crude_A': 400,
    'crude_B': 350,
    'crude_C': 300
}

# Yield patterns (volume fraction of each product from each crude)
yield_pattern = {
    ('crude_A', 'gasoline'): 0.45,    # 45% gasoline yield from crude A
    ('crude_A', 'diesel'): 0.35,      # 35% diesel yield from crude A
    ('crude_A', 'fuel_oil'): 0.20,    # 20% fuel oil yield from crude A
    
    ('crude_B', 'gasoline'): 0.35,    # 35% gasoline yield from crude B
    ('crude_B', 'diesel'): 0.45,      # 45% diesel yield from crude B
    ('crude_B', 'fuel_oil'): 0.20,    # 20% fuel oil yield from crude B
    
    ('crude_C', 'gasoline'): 0.50,    # 50% gasoline yield from crude C
    ('crude_C', 'diesel'): 0.30,      # 30% diesel yield from crude C
    ('crude_C', 'fuel_oil'): 0.20,    # 20% fuel oil yield from crude C
}

# Input streams properties (sulfur content in weight fraction)
sulfur_content = {
    'crude_A': 0.015,
    'crude_B': 0.020,
    'crude_C': 0.010
}

# Product specifications (maximum sulfur content)
max_sulfur = {
    'gasoline': 0.012,
    'diesel': 0.018,
    'fuel_oil': 0.035
}

# Add processing costs for each blending unit ($/barrel)
processing_costs = {
    'blender_1': 8,    # Processing cost for blender 1
    'blender_2': 10    # Processing cost for blender 2 (slightly higher due to different technology/age)
}

# Maximum treatment capacity for each blending unit (barrels per day)
blender_capacity = {
    'blender_1': 400,  # Maximum capacity for blender 1
    'blender_2': 350   # Maximum capacity for blender 2
}

# Product selling prices ($/barrel)
selling_prices = {
    'gasoline': 125,
    'diesel': 115,
    'fuel_oil': 95
}

# Transportation costs ($ per barrel)
transport_to_warehouse = {
    ('blender_1', 'warehouse_1'): 2,
    ('blender_1', 'warehouse_2'): 3,
    ('blender_2', 'warehouse_1'): 2.5,
    ('blender_2', 'warehouse_2'): 2.8
}

transport_to_market = {
    ('warehouse_1', 'market_1'): 3,
    ('warehouse_1', 'market_2'): 4,
    ('warehouse_1', 'market_3'): 4.5,
    ('warehouse_2', 'market_1'): 4,
    ('warehouse_2', 'market_2'): 3,
    ('warehouse_2', 'market_3'): 3.5
}

# Define variables
blend_vars = LpVariable.dicts("blend",
                            ((i, b) for i in input_streams for b in blending_units),
                            lowBound=0,
                            cat='Integer')

prod_vars = LpVariable.dicts("production",
                           ((b, p) for b in blending_units for p in products),
                           lowBound=0,
                           cat='Integer')

warehouse_vars = LpVariable.dicts("to_warehouse",
                                ((b, w, p) for b in blending_units 
                                for w in temp_warehouses 
                                for p in products),
                                lowBound=0,
                                cat='Integer')

market_vars = LpVariable.dicts("to_market",
                             ((w, m, p) for w in temp_warehouses 
                             for m in selling_points 
                             for p in products),
                             lowBound=0,
                             cat='Integer')

# Objective function
prob += (
    lpSum(market_vars[w,m,p] * selling_prices[p] 
          for w in temp_warehouses 
          for m in selling_points 
          for p in products) -
    lpSum(blend_vars[i,b] * input_costs[i] 
          for i in input_streams 
          for b in blending_units) -
    lpSum(blend_vars[i,b] * processing_costs[b]  # Added processing costs
          for i in input_streams 
          for b in blending_units) -
    lpSum(warehouse_vars[b,w,p] * transport_to_warehouse[b,w] 
          for b in blending_units 
          for w in temp_warehouses 
          for p in products) -
    lpSum(market_vars[w,m,p] * transport_to_market[w,m] 
          for w in temp_warehouses 
          for m in selling_points 
          for p in products)
)

# Constraints
# 1. Product yield from crude oils in blending units
for b in blending_units:
    for p in products:
        prob += prod_vars[b,p] == lpSum(blend_vars[i,b] * yield_pattern[i,p] 
                                      for i in input_streams)

# 2. Sulfur content specifications
for b in blending_units:
    for p in products:
        prob += (lpSum(blend_vars[i,b] * yield_pattern[i,p] * sulfur_content[i] 
                for i in input_streams) <= 
                max_sulfur[p] * lpSum(blend_vars[i,b] * yield_pattern[i,p] 
                for i in input_streams))

# 3. Mass balance at warehouses
for b in blending_units:
    for p in products:
        prob += prod_vars[b,p] == \
                lpSum(warehouse_vars[b,w,p] for w in temp_warehouses)

# 4. Mass balance at warehouses to markets
for w in temp_warehouses:
    for p in products:
        prob += lpSum(warehouse_vars[b,w,p] for b in blending_units) == \
                lpSum(market_vars[w,m,p] for m in selling_points)

# Capacity constraints
# Blender capacity (barrels per day)
for b in blending_units:
    prob += lpSum(blend_vars[i,b] for i in input_streams) <= blender_capacity[b]

# Warehouse capacity (barrels)
for w in temp_warehouses:
    prob += lpSum(warehouse_vars[b,w,p] 
                 for b in blending_units 
                 for p in products) <= 600

# Market demand (minimum and maximum)
for m in selling_points:
    for p in products:
        prob += lpSum(market_vars[w,m,p] for w in temp_warehouses) >= 20
        prob += lpSum(market_vars[w,m,p] for w in temp_warehouses) <= 150

# Input stream availability constraints
for i in input_streams:
    # Total usage across all blenders cannot exceed availability
    prob += lpSum(blend_vars[i,b] for b in blending_units) <= crude_availability[i]

# Solve the problem
prob.solve()

# Print results
print(f"Status: {LpStatus[prob.status]}")
print(f"Optimal Value: ${value(prob.objective):,.2f}")

# Print detailed results
print("\nCrude Oil Processing:")
for b in blending_units:
    for i in input_streams:
        if value(blend_vars[i,b]) > 0:
            print(f"{i} to {b}: {value(blend_vars[i,b]):,.2f} barrels")

print("\nProduct Production:")
for b in blending_units:
    for p in products:
        if value(prod_vars[b,p]) > 0:
            print(f"{p} from {b}: {value(prod_vars[b,p]):,.2f} barrels")
            
print("\nProduct Distribution:")
for w in temp_warehouses:
    for m in selling_points:
        for p in products:
            if value(market_vars[w,m,p]) > 0:
                print(f"{p} from {w} to {m}: {value(market_vars[w,m,p]):,.2f} barrels")
                
print("\nProcessing Costs Summary:")
for b in blending_units:
    total_processed = sum(value(blend_vars[i,b]) for i in input_streams)
    if total_processed > 0:
        cost = total_processed * processing_costs[b]
        print(f"{b} processed {total_processed:,.2f} barrels at ${processing_costs[b]}/barrel = ${cost:,.2f}")

