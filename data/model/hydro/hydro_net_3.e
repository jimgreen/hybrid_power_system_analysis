<HydroMedium>
@ density compressibility molar_mass temperature flow_factor
# 0.08375 1.0 0.002016 288.15 0.35
</HydroMedium>

<HydroNode>
@ idx name pressure run_stat
# 1 hydro_n1 3.00 1
# 2 hydro_n2 2.95 1
# 3 hydro_n3 2.98 1
# 4 hydro_n4 2.90 1
# 5 hydro_n5 2.85 1
</HydroNode>

<HydroSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 hydro_source 1 PRESSURE 3.00 0.0 1.0 0.0 2.0 1
</HydroSource>

<HydroLoad>
@ idx name node flow_set run_stat
# 1 hydro_load_2 2 0.10 1
# 2 hydro_load_4 4 0.15 1
# 3 hydro_load_5 5 0.25 1
</HydroLoad>

<HydroPipe>
@ idx name i_node j_node conductance run_stat
# 1 hydro_pipe_12 1 2 3.0 1
# 2 hydro_pipe_45 4 5 2.5 1
</HydroPipe>

<HydroValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 hydro_valve_34 3 4 OPEN 2.8 0.0 1
</HydroValve>

<HydroCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 hydro_compressor_23 2 3 RATIO 1.01 0.4 1
</HydroCompressor>
