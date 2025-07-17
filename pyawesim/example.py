from bindings import *
import numpy as np

sim = sim_malloc()
city_width = 1000   # 1km
num_cars = 256        # 8 cars (incl. agent)
init_clock = clock_reading(0, 8, 0, 0)   # Monday 8:00 AM
init_weather = WEATHER_SUNNY   
awesim_setup(sim, city_width, num_cars, 0.02, init_clock, init_weather)  # set's up sim variables, awesim map and adds cars randomly 
sim_set_agent_enabled(sim, True)         # car 0 won't be NPC  
decision_interval = 0.1
total_play_time = 1000.0
agent = sim_get_agent_car(sim)  
sim_set_synchronized(sim, True, 1.0)  # synchronize simulation with real time  
sim_connect_to_render_server(sim, "127.0.0.1", 4242)  # before this, start the server with `../bin/awesim_render_server 4242 --persistent`  

current_procedure = PROCEDURE_CRUISE
procedure = sim_get_ongoing_procedure(sim, agent.id)  # get the ongoing procedure for the agent
status = PROCEDURE_STATUS_NONE

print(f"Running simulation for {total_play_time} seconds with decision interval of {decision_interval} seconds. Press Ctrl+C to stop.")
while sim_get_time(sim) < total_play_time:  
    situational_awareness_build(sim, agent.id)  # holds most of the state variables needed for decision making  
    situation = sim_get_situational_awareness(sim, agent.id)  # get situational awareness for the agent  
    distance_to_lead_vehicle = situation.distance_to_lead_vehicle   # example variable
    speed = car_get_speed(agent)

    if status != PROCEDURE_STATUS_IN_PROGRESS and status != PROCEDURE_STATUS_INITIALIZED:
        if current_procedure == PROCEDURE_CRUISE:
            print("Initializing CRUISE procedure...")
            args = np.array([5.0, 13.0, 1.0, 2.0, 1.0], dtype=np.float64)  # cruise speed, follow distance, adaptive cruise control, preferred acceleration profile
            c_ptr = args.__array_interface__['data'][0]
            status = procedure_init(sim, agent, procedure, PROCEDURE_CRUISE, c_ptr)
        elif current_procedure == PROCEDURE_MERGE:
            print("Initializing MERGE procedure...")
            merge_direction = INDICATOR_RIGHT if sim_get_time(sim) % 10 < 5 else INDICATOR_LEFT
            args = np.array([float(merge_direction), 5.0, 13.0, 1.0, 2.0, 1.0], dtype=np.float64)
            c_ptr = args.__array_interface__['data'][0]
            status = procedure_init(sim, agent, procedure, PROCEDURE_MERGE, c_ptr)

        if status != PROCEDURE_STATUS_INITIALIZED:
            print(f"Agent {agent.id} failed to initialize procedure {current_procedure}. Status: {status}. Switching to default control for this step.\n")
            status = PROCEDURE_STATUS_NONE
    
    if status != PROCEDURE_STATUS_NONE:
        status = procedure_step(sim, agent, procedure)  # advance the procedure
        if status != PROCEDURE_STATUS_IN_PROGRESS and status != PROCEDURE_STATUS_COMPLETED:
            print(f"Agent {agent.id} failed to execute procedure {current_procedure}. Status: {status}\n")
        if status != PROCEDURE_STATUS_IN_PROGRESS:
            current_procedure = PROCEDURE_MERGE if current_procedure == PROCEDURE_CRUISE else PROCEDURE_CRUISE
    else:
        print(f"Agent {agent.id} is not executing any procedure. Current status: {status}\n")
        car_reset_all_control_variables(agent)  # reset all control variables to default values

    simulate(sim, decision_interval)    # simulate 0.1 seconds  
print("Simulation finished.")

sim_disconnect_from_render_server(sim)  # close rendering window  
sim_free(sim)  
