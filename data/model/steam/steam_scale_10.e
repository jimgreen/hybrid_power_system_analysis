<SteamMedium>
@ density compressibility molar_mass temperature heat_capacity ambient_enthalpy reference_temperature reference_enthalpy feedwater_enthalpy flow_factor
# 4.0 1.0 0.018 473.15 2.08 419.0 100.0 2676.0 419.0 1.0
</SteamMedium>

<SteamNode>
@ idx name pressure enthalpy temperature run_stat
# 1 steam_n1 10 3000.0 255.769230769 1
# 2 steam_n2 9.985 3000.0 255.769230769 1
# 3 steam_n3 9.985 3000.0 255.769230769 1
# 4 steam_n4 9.97 3000.0 255.769230769 1
# 5 steam_n5 9.97 3000.0 255.769230769 1
# 6 steam_n6 9.97 3000.0 255.769230769 1
# 7 steam_n7 9.97 3000.0 255.769230769 1
# 8 steam_n8 9.955 3000.0 255.769230769 1
# 9 steam_n9 9.955 3000.0 255.769230769 1
# 10 steam_n10 9.955 3000.0 255.769230769 1
</SteamNode>

<SteamSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max enthalpy_set run_stat
# 1 steam_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 3000.0 1
</SteamSource>

<SteamLoad>
@ idx name node flow_set condensate_enthalpy run_stat
# 1 steam_load_1 6 0.20000000000000001 419.0 1
# 2 steam_load_2 7 0.20000000000000001 419.0 1
# 3 steam_load_3 8 0.20000000000000001 419.0 1
# 4 steam_load_4 9 0.20000000000000001 419.0 1
# 5 steam_load_5 10 0.20000000000000001 419.0 1
</SteamLoad>

<SteamPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 steam_pipe_2_4 2 4 1.0 1.9999999999999999e-06 1
# 2 steam_pipe_2_5 2 5 1.0 3.0000000000000001e-06 1
# 3 steam_pipe_3_6 3 6 1.0 9.9999999999999995e-07 1
# 4 steam_pipe_3_7 3 7 1.0 1.9999999999999999e-06 1
# 5 steam_pipe_4_8 4 8 1.0 3.0000000000000001e-06 1
# 6 steam_pipe_4_9 4 9 1.0 9.9999999999999995e-07 1
# 7 steam_pipe_5_10 5 10 1.0 1.9999999999999999e-06 1
</SteamPipe>

<SteamValve>
@ idx name i_node j_node control_type conductance flow_set heat_loss run_stat
# 1 steam_valve_1_3 1 3 OPEN 1.0 0.0 9.9999999999999995e-07 1
</SteamValve>

<SteamPressureReducer>
@ idx name i_node j_node control_type ratio flow_set heat_loss run_stat
# 1 steam_reducer_1_2 1 2 RATIO 1.0 0.0 3.0000000000000001e-06 1
</SteamPressureReducer>
