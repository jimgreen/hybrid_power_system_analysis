<GasMedium>
@ density compressibility molar_mass temperature flow_factor
# 0.8 1.0 0.018 288.15 1.0
</GasMedium>

<GasNode>
@ idx name pressure run_stat
# 1 gas_n1 10 1
# 2 gas_n2 9.985 1
# 3 gas_n3 9.985 1
# 4 gas_n4 9.97 1
# 5 gas_n5 9.97 1
# 6 gas_n6 9.97 1
# 7 gas_n7 9.97 1
# 8 gas_n8 9.955 1
# 9 gas_n9 9.955 1
# 10 gas_n10 9.955 1
</GasNode>

<GasSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 gas_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 1
</GasSource>

<GasLoad>
@ idx name node flow_set run_stat
# 1 gas_load_1 6 0.20000000000000001 1
# 2 gas_load_2 7 0.20000000000000001 1
# 3 gas_load_3 8 0.20000000000000001 1
# 4 gas_load_4 9 0.20000000000000001 1
# 5 gas_load_5 10 0.20000000000000001 1
</GasLoad>

<GasPipe>
@ idx name i_node j_node conductance run_stat
# 1 gas_pipe_2_4 2 4 1.0 1
# 2 gas_pipe_2_5 2 5 1.0 1
# 3 gas_pipe_3_6 3 6 1.0 1
# 4 gas_pipe_3_7 3 7 1.0 1
# 5 gas_pipe_4_8 4 8 1.0 1
# 6 gas_pipe_4_9 4 9 1.0 1
# 7 gas_pipe_5_10 5 10 1.0 1
</GasPipe>

<GasValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 gas_valve_1_3 1 3 OPEN 1.0 0.0 1
</GasValve>

<GasCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 gas_compressor_1_2 1 2 RATIO 1.0 0.0 1
</GasCompressor>
