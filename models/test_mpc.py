from control.mpc import run_mpc


params = {

    "mu_max": 0.5,

    "Ks": 1.0,

    "KDO": 0.001,

    "Yxs": 0.5,

    "Yps": 0.8,

    "qP_max": 0.05,

    "qO2": 0.2,

    "Csat": 0.008,

    "hill": 2,

    "half_light": 30,

    "basal": 0.1,

    "max_light": 100,

    "min_kla": 20,

    "max_kla": 150,

    "DO_target": 0.002,

    "weight_light": 0.01,

    "weight_kla": 0.01,

    "weight_oxygen": 100

}


# Initial reactor conditions:
# Biomass, Substrate, Dissolved Oxygen, Product

initial_state = [

    0.1,

    20.0,

    0.007,

    0.0

]


result = run_mpc(

    initial_state=initial_state,

    params=params,

    duration=20,

    control_interval=0.5,

    prediction_horizon=2.0

)


print("\nMODEL PREDICTIVE CONTROL RESULTS")

print("--------------------------------------")

print("Final biomass:", result["biomass"][-1], "g/L")

print("Final substrate:", result["substrate"][-1], "g/L")

print("Final dissolved oxygen:", result["oxygen"][-1], "g/L")

print("Final product:", result["product"][-1], "g/L")

print("\nFINAL CONTROL SETTINGS")

print("--------------------------------------")

print("Light intensity:", result["light"][-1])

print("Oxygen transfer coefficient:", result["kla"][-1])

print("\nAVERAGE CONTROL SETTINGS")

print("--------------------------------------")

print("Average light:", result["light"].mean())

print("Average kLa:", result["kla"].mean())