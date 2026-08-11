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
# 11 hydro_n11 9.955 1
# 12 hydro_n12 9.955 1
# 13 hydro_n13 9.955 1
# 14 hydro_n14 9.955 1
# 15 hydro_n15 9.955 1
# 16 hydro_n16 9.94 1
# 17 hydro_n17 9.94 1
# 18 hydro_n18 9.94 1
# 19 hydro_n19 9.94 1
# 20 hydro_n20 9.94 1
# 21 hydro_n21 9.94 1
# 22 hydro_n22 9.94 1
# 23 hydro_n23 9.94 1
# 24 hydro_n24 9.94 1
# 25 hydro_n25 9.94 1
# 26 hydro_n26 9.94 1
# 27 hydro_n27 9.94 1
# 28 hydro_n28 9.94 1
# 29 hydro_n29 9.94 1
# 30 hydro_n30 9.94 1
# 31 hydro_n31 9.94 1
# 32 hydro_n32 9.925 1
# 33 hydro_n33 9.925 1
# 34 hydro_n34 9.925 1
# 35 hydro_n35 9.925 1
# 36 hydro_n36 9.925 1
# 37 hydro_n37 9.925 1
# 38 hydro_n38 9.925 1
# 39 hydro_n39 9.925 1
# 40 hydro_n40 9.925 1
# 41 hydro_n41 9.925 1
# 42 hydro_n42 9.925 1
# 43 hydro_n43 9.925 1
# 44 hydro_n44 9.925 1
# 45 hydro_n45 9.925 1
# 46 hydro_n46 9.925 1
# 47 hydro_n47 9.925 1
# 48 hydro_n48 9.925 1
# 49 hydro_n49 9.925 1
# 50 hydro_n50 9.925 1
</HydroNode>

<HydroSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 hydro_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 1
</HydroSource>

<HydroLoad>
@ idx name node flow_set run_stat
# 1 hydro_load_1 26 0.040000000000000001 1
# 2 hydro_load_2 27 0.040000000000000001 1
# 3 hydro_load_3 28 0.040000000000000001 1
# 4 hydro_load_4 29 0.040000000000000001 1
# 5 hydro_load_5 30 0.040000000000000001 1
# 6 hydro_load_6 31 0.040000000000000001 1
# 7 hydro_load_7 32 0.040000000000000001 1
# 8 hydro_load_8 33 0.040000000000000001 1
# 9 hydro_load_9 34 0.040000000000000001 1
# 10 hydro_load_10 35 0.040000000000000001 1
# 11 hydro_load_11 36 0.040000000000000001 1
# 12 hydro_load_12 37 0.040000000000000001 1
# 13 hydro_load_13 38 0.040000000000000001 1
# 14 hydro_load_14 39 0.040000000000000001 1
# 15 hydro_load_15 40 0.040000000000000001 1
# 16 hydro_load_16 41 0.040000000000000001 1
# 17 hydro_load_17 42 0.040000000000000001 1
# 18 hydro_load_18 43 0.040000000000000001 1
# 19 hydro_load_19 44 0.040000000000000001 1
# 20 hydro_load_20 45 0.040000000000000001 1
# 21 hydro_load_21 46 0.040000000000000001 1
# 22 hydro_load_22 47 0.040000000000000001 1
# 23 hydro_load_23 48 0.040000000000000001 1
# 24 hydro_load_24 49 0.040000000000000001 1
# 25 hydro_load_25 50 0.040000000000000001 1
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
# 8 hydro_pipe_5_11 5 11 1.0 1
# 9 hydro_pipe_6_12 6 12 1.0 1
# 10 hydro_pipe_6_13 6 13 1.0 1
# 11 hydro_pipe_7_14 7 14 1.0 1
# 12 hydro_pipe_7_15 7 15 1.0 1
# 13 hydro_pipe_8_16 8 16 1.0 1
# 14 hydro_pipe_9_18 9 18 1.0 1
# 15 hydro_pipe_9_19 9 19 1.0 1
# 16 hydro_pipe_10_20 10 20 1.0 1
# 17 hydro_pipe_10_21 10 21 1.0 1
# 18 hydro_pipe_11_22 11 22 1.0 1
# 19 hydro_pipe_11_23 11 23 1.0 1
# 20 hydro_pipe_12_24 12 24 1.0 1
# 21 hydro_pipe_12_25 12 25 1.0 1
# 22 hydro_pipe_13_26 13 26 1.0 1
# 23 hydro_pipe_13_27 13 27 1.0 1
# 24 hydro_pipe_14_28 14 28 1.0 1
# 25 hydro_pipe_14_29 14 29 1.0 1
# 26 hydro_pipe_15_30 15 30 1.0 1
# 27 hydro_pipe_15_31 15 31 1.0 1
# 28 hydro_pipe_16_32 16 32 1.0 1
# 29 hydro_pipe_16_33 16 33 1.0 1
# 30 hydro_pipe_17_35 17 35 1.0 1
# 31 hydro_pipe_18_36 18 36 1.0 1
# 32 hydro_pipe_18_37 18 37 1.0 1
# 33 hydro_pipe_19_38 19 38 1.0 1
# 34 hydro_pipe_19_39 19 39 1.0 1
# 35 hydro_pipe_20_40 20 40 1.0 1
# 36 hydro_pipe_21_42 21 42 1.0 1
# 37 hydro_pipe_21_43 21 43 1.0 1
# 38 hydro_pipe_22_44 22 44 1.0 1
# 39 hydro_pipe_22_45 22 45 1.0 1
# 40 hydro_pipe_23_46 23 46 1.0 1
# 41 hydro_pipe_23_47 23 47 1.0 1
# 42 hydro_pipe_24_48 24 48 1.0 1
# 43 hydro_pipe_24_49 24 49 1.0 1
# 44 hydro_pipe_25_50 25 50 1.0 1
</HydroPipe>

<HydroValve>
@ idx name i_node j_node control_type conductance flow_set run_stat
# 1 hydro_valve_1_3 1 3 OPEN 1.0 0.0 1
# 2 hydro_valve_8_17 8 17 OPEN 1.0 0.0 1
# 3 hydro_valve_17_34 17 34 OPEN 1.0 0.0 1
</HydroValve>

<HydroCompressor>
@ idx name i_node j_node control_type ratio flow_set run_stat
# 1 hydro_compressor_1_2 1 2 RATIO 1.0 0.0 1
# 2 hydro_compressor_20_41 20 41 RATIO 1.0 0.0 1
</HydroCompressor>
