<HydroMedium>
@ density compressibility molar_mass temperature flow_factor
# 0.08375 1.0 0.002016 288.15 1.0
</HydroMedium>

<HydroNode>
@ idx name pressure run_stat
# 1 hydro_n1 10 1
# 2 hydro_n2 9.985 1
# 3 hydro_n3 9.985 1
# 4 hydro_n4 9.97 1
# 5 hydro_n5 9.97 1
# 6 hydro_n6 9.97 1
# 7 hydro_n7 9.97 1
# 8 hydro_n8 9.955 1
# 9 hydro_n9 9.955 1
# 10 hydro_n10 9.955 1
</HydroNode>

<HydroSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 hydro_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 1
</HydroSource>

<HydroLoad>
@ idx name node flow_set run_stat
# 1 hydro_load_1 6 0.20000000000000001 1
# 2 hydro_load_2 7 0.20000000000000001 1
# 3 hydro_load_3 8 0.20000000000000001 1
# 4 hydro_load_4 9 0.20000000000000001 1
# 5 hydro_load_5 10 0.20000000000000001 1
</HydroLoad>

<HydroPipe>
@ idx name i_node j_node conductance run_stat
# 1 hydro_pipe_2_4 2 4 1.0 1
# 2 hydro_pipe_2_5 2 5 1.0 1
# 3 hydro_pipe_3_6 3 6 1.0 1
# 4 hydro_pipe_3_7 3 7 1.0 1
# 5 hydro_pipe_4_8 4 8 1.0 1
# 6 hydro_pipe_4_9 4 9 1.0 1
# 7 hydro_pipe_5_10 5 10 1.0 1
</HydroPipe>

<HydroValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 hydro_valve_1_3 1 3 OPEN 1.0 0.0 1
</HydroValve>

<HydroCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 hydro_compressor_1_2 1 2 RATIO 1.0 0.0 1
</HydroCompressor>
