<GasMedium>
@ density compressibility molar_mass temperature flow_factor
# 0.8 0.9 0.018 288.15 1.0
</GasMedium>

<GasNode>
@ idx name pressure run_stat
# 1 gas_n1 5.00 1
# 2 gas_n2 4.90 1
# 3 gas_n3 4.95 1
# 4 gas_n4 4.80 1
# 5 gas_n5 4.75 1
</GasNode>

<GasSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 gas_source 1 PRESSURE 5.00 0.0 1.0 0.0 3.0 1
</GasSource>

<GasLoad>
@ idx name node flow_set run_stat
# 1 gas_load_2 2 0.2 1
# 2 gas_load_4 4 0.3 1
# 3 gas_load_5 5 0.5 1
</GasLoad>

<GasPipe>
@ idx name i_node j_node conductance run_stat
# 1 gas_pipe_12 1 2 1.0 1
# 2 gas_pipe_45 4 5 1.0 1
</GasPipe>

<GasValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 gas_valve_34 3 4 OPEN 1.0 0.0 1
</GasValve>

<GasCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 gas_compressor_23 2 3 RATIO 1.02 0.8 1
</GasCompressor>
