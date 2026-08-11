<SteamMedium>
@ density compressibility molar_mass temperature flow_factor heat_capacity ambient_temperature reference_temperature reference_enthalpy ambient_enthalpy feedwater_enthalpy
# 4.0 1.0 0.018 500.0 1.0 2.08 25.0 100.0 2676.0 419.0 419.0
</SteamMedium>

<SteamNode>
@ idx name pressure enthalpy run_stat
# 1 steam_n1 5.00 3200.0 1
# 2 steam_n2 4.90 3145.0 1
# 3 steam_n3 4.41 3110.0 1
# 4 steam_n4 4.34 3060.0 1
# 5 steam_n5 4.30 2960.0 1
</SteamNode>

<SteamSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max enthalpy_set run_stat
# 1 steam_source 1 PRESSURE 5.00 0.0 1.0 0.0 3.0 3200.0 1
</SteamSource>

<SteamLoad>
@ idx name node flow_set condensate_enthalpy run_stat
# 1 steam_load_2 2 0.2 419.0 1
# 2 steam_load_4 4 0.3 419.0 1
# 3 steam_load_5 5 0.5 419.0 1
</SteamLoad>

<SteamPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 steam_pipe_12 1 2 1.0 0.02 1
# 2 steam_pipe_45 4 5 1.0 0.02 1
</SteamPipe>

<SteamValve>
@ idx name i_node j_node control_type conductance flow_set heat_loss run_stat
# 1 steam_valve_34 3 4 OPEN 1.0 0.0 0.015 1
</SteamValve>

<SteamPressureReducer>
@ idx name i_node j_node control_type ratio flow_set heat_loss run_stat
# 1 steam_reducer_23 2 3 RATIO 0.90 0.8 0.01 1
</SteamPressureReducer>
