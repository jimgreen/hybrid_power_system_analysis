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
# 11 gas_n11 9.955 1
# 12 gas_n12 9.955 1
# 13 gas_n13 9.955 1
# 14 gas_n14 9.955 1
# 15 gas_n15 9.955 1
# 16 gas_n16 9.94 1
# 17 gas_n17 9.94 1
# 18 gas_n18 9.94 1
# 19 gas_n19 9.94 1
# 20 gas_n20 9.94 1
# 21 gas_n21 9.94 1
# 22 gas_n22 9.94 1
# 23 gas_n23 9.94 1
# 24 gas_n24 9.94 1
# 25 gas_n25 9.94 1
# 26 gas_n26 9.94 1
# 27 gas_n27 9.94 1
# 28 gas_n28 9.94 1
# 29 gas_n29 9.94 1
# 30 gas_n30 9.94 1
# 31 gas_n31 9.94 1
# 32 gas_n32 9.925 1
# 33 gas_n33 9.925 1
# 34 gas_n34 9.925 1
# 35 gas_n35 9.925 1
# 36 gas_n36 9.925 1
# 37 gas_n37 9.925 1
# 38 gas_n38 9.925 1
# 39 gas_n39 9.925 1
# 40 gas_n40 9.925 1
# 41 gas_n41 9.925 1
# 42 gas_n42 9.925 1
# 43 gas_n43 9.925 1
# 44 gas_n44 9.925 1
# 45 gas_n45 9.925 1
# 46 gas_n46 9.925 1
# 47 gas_n47 9.925 1
# 48 gas_n48 9.925 1
# 49 gas_n49 9.925 1
# 50 gas_n50 9.925 1
</GasNode>

<GasSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 gas_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 1
</GasSource>

<GasLoad>
@ idx name node flow_set run_stat
# 1 gas_load_1 26 0.040000000000000001 1
# 2 gas_load_2 27 0.040000000000000001 1
# 3 gas_load_3 28 0.040000000000000001 1
# 4 gas_load_4 29 0.040000000000000001 1
# 5 gas_load_5 30 0.040000000000000001 1
# 6 gas_load_6 31 0.040000000000000001 1
# 7 gas_load_7 32 0.040000000000000001 1
# 8 gas_load_8 33 0.040000000000000001 1
# 9 gas_load_9 34 0.040000000000000001 1
# 10 gas_load_10 35 0.040000000000000001 1
# 11 gas_load_11 36 0.040000000000000001 1
# 12 gas_load_12 37 0.040000000000000001 1
# 13 gas_load_13 38 0.040000000000000001 1
# 14 gas_load_14 39 0.040000000000000001 1
# 15 gas_load_15 40 0.040000000000000001 1
# 16 gas_load_16 41 0.040000000000000001 1
# 17 gas_load_17 42 0.040000000000000001 1
# 18 gas_load_18 43 0.040000000000000001 1
# 19 gas_load_19 44 0.040000000000000001 1
# 20 gas_load_20 45 0.040000000000000001 1
# 21 gas_load_21 46 0.040000000000000001 1
# 22 gas_load_22 47 0.040000000000000001 1
# 23 gas_load_23 48 0.040000000000000001 1
# 24 gas_load_24 49 0.040000000000000001 1
# 25 gas_load_25 50 0.040000000000000001 1
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
# 8 gas_pipe_5_11 5 11 1.0 1
# 9 gas_pipe_6_12 6 12 1.0 1
# 10 gas_pipe_6_13 6 13 1.0 1
# 11 gas_pipe_7_14 7 14 1.0 1
# 12 gas_pipe_7_15 7 15 1.0 1
# 13 gas_pipe_8_16 8 16 1.0 1
# 14 gas_pipe_9_18 9 18 1.0 1
# 15 gas_pipe_9_19 9 19 1.0 1
# 16 gas_pipe_10_20 10 20 1.0 1
# 17 gas_pipe_10_21 10 21 1.0 1
# 18 gas_pipe_11_22 11 22 1.0 1
# 19 gas_pipe_11_23 11 23 1.0 1
# 20 gas_pipe_12_24 12 24 1.0 1
# 21 gas_pipe_12_25 12 25 1.0 1
# 22 gas_pipe_13_26 13 26 1.0 1
# 23 gas_pipe_13_27 13 27 1.0 1
# 24 gas_pipe_14_28 14 28 1.0 1
# 25 gas_pipe_14_29 14 29 1.0 1
# 26 gas_pipe_15_30 15 30 1.0 1
# 27 gas_pipe_15_31 15 31 1.0 1
# 28 gas_pipe_16_32 16 32 1.0 1
# 29 gas_pipe_16_33 16 33 1.0 1
# 30 gas_pipe_17_35 17 35 1.0 1
# 31 gas_pipe_18_36 18 36 1.0 1
# 32 gas_pipe_18_37 18 37 1.0 1
# 33 gas_pipe_19_38 19 38 1.0 1
# 34 gas_pipe_19_39 19 39 1.0 1
# 35 gas_pipe_20_40 20 40 1.0 1
# 36 gas_pipe_21_42 21 42 1.0 1
# 37 gas_pipe_21_43 21 43 1.0 1
# 38 gas_pipe_22_44 22 44 1.0 1
# 39 gas_pipe_22_45 22 45 1.0 1
# 40 gas_pipe_23_46 23 46 1.0 1
# 41 gas_pipe_23_47 23 47 1.0 1
# 42 gas_pipe_24_48 24 48 1.0 1
# 43 gas_pipe_24_49 24 49 1.0 1
# 44 gas_pipe_25_50 25 50 1.0 1
</GasPipe>

<GasValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 gas_valve_1_3 1 3 OPEN 1.0 0.0 1
# 2 gas_valve_8_17 8 17 OPEN 1.0 0.0 1
# 3 gas_valve_17_34 17 34 OPEN 1.0 0.0 1
</GasValve>

<GasCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 gas_compressor_1_2 1 2 RATIO 1.0 0.0 1
# 2 gas_compressor_20_41 20 41 RATIO 1.0 0.0 1
</GasCompressor>
