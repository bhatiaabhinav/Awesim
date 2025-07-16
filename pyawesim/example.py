from bindings import *

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

current_procedure = "cruise"
merge_direction = 'r'  # default merge direction

print(f"Running simulation for {total_play_time} seconds with decision interval of {decision_interval} seconds. Press Ctrl+C to stop.")
while sim_get_time(sim) < total_play_time:  
    situational_awareness_build(sim, agent.id)  # holds most of the state variables needed for decision making  
    situation = sim_get_situational_awareness(sim, agent.id)  # get situational awareness for the agent  
    distance_to_lead_vehicle = situation.distance_to_lead_vehicle   # example variable
    speed = car_get_speed(agent)

    if current_procedure == "cruise":
        print(f"trying to cruise. current speed = {to_mph(speed)} mph.\n")
        cruising_completed = cruise(agent, from_mph(30))
        if cruising_completed:
            print(f"Agent {agent.id} is cruising at 30 mph.\n")
            current_procedure = "merge"
    elif current_procedure == "merge":
        # every 5 seconds, change merging direction:
        print(f"trying to merge {merge_direction}. current speed = {to_mph(speed)} mph.\n")
        merging_completed = merge(agent, sim, merge_direction, 5.0) # try to merge in the given direction with a 5 second duration
        if merging_completed:
            print("Merging completed")
            current_procedure = "cruise"
            merge_direction = 'r' if sim_get_time(sim) % 10 < 5 else 'l'    # for next merge

    simulate(sim, decision_interval)    # simulate 0.1 seconds  
print("Simulation finished.")

sim_disconnect_from_render_server(sim)  # close rendering window  
sim_free(sim)  
