<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure temperature supply_temperature return_temperature run_stat
# 1 heat_supply_n1 10 90.0 90.0 70.0 1
# 2 heat_supply_n2 9.99 90.0 90.0 70.0 1
# 3 heat_supply_n3 9.99 90.0 90.0 70.0 1
# 4 heat_supply_n4 9.98 90.0 90.0 70.0 1
# 5 heat_supply_n5 9.98 90.0 90.0 70.0 1
# 6 heat_supply_n6 9.98 90.0 90.0 70.0 1
# 7 heat_supply_n7 9.98 90.0 90.0 70.0 1
# 8 heat_supply_n8 9.97 90.0 90.0 70.0 1
# 9 heat_supply_n9 9.97 90.0 90.0 70.0 1
# 10 heat_supply_n10 9.97 90.0 90.0 70.0 1
# 11 heat_supply_n11 9.97 90.0 90.0 70.0 1
# 12 heat_supply_n12 9.97 90.0 90.0 70.0 1
# 13 heat_return_n1 5 70.0 90.0 70.0 1
# 14 heat_return_n2 5.01 70.0 90.0 70.0 1
# 15 heat_return_n3 5.01 70.0 90.0 70.0 1
# 16 heat_return_n4 5.02 70.0 90.0 70.0 1
# 17 heat_return_n5 5.02 70.0 90.0 70.0 1
# 18 heat_return_n6 5.02 70.0 90.0 70.0 1
# 19 heat_return_n7 5.02 70.0 90.0 70.0 1
# 20 heat_return_n8 5.03 70.0 90.0 70.0 1
# 21 heat_return_n9 5.03 70.0 90.0 70.0 1
# 22 heat_return_n10 5.03 70.0 90.0 70.0 1
# 23 heat_return_n11 5.03 70.0 90.0 70.0 1
# 24 heat_return_n12 5.03 70.0 90.0 70.0 1
# 25 heat_secondary_n1 8 82.0 82.0 62.0 1
# 26 heat_secondary_n2 7.99 82.0 82.0 62.0 1
# 27 heat_secondary_n3 7.99 82.0 82.0 62.0 1
# 28 heat_secondary_n4 7.98 82.0 82.0 62.0 1
# 29 heat_secondary_n5 7.98 82.0 82.0 62.0 1
# 30 heat_secondary_n6 7.98 82.0 82.0 62.0 1
# 31 heat_secondary_n7 7.98 82.0 82.0 62.0 1
# 32 heat_secondary_n8 7.97 82.0 82.0 62.0 1
# 33 heat_secondary_n9 7.97 82.0 82.0 62.0 1
# 34 heat_secondary_n10 7.97 82.0 82.0 62.0 1
# 35 heat_secondary_n11 7.97 82.0 82.0 62.0 1
# 36 heat_secondary_n12 7.97 82.0 82.0 62.0 1
# 37 heat_secondary_n13 7.97 82.0 82.0 62.0 1
# 38 heat_secondary_n14 7.97 82.0 82.0 62.0 1
# 39 heat_secondary_n15 7.97 82.0 82.0 62.0 1
# 40 heat_secondary_n16 7.96 82.0 82.0 62.0 1
# 41 heat_secondary_n17 7.96 82.0 82.0 62.0 1
# 42 heat_secondary_n18 7.96 82.0 82.0 62.0 1
# 43 heat_secondary_n19 7.96 82.0 82.0 62.0 1
# 44 heat_secondary_n20 7.96 82.0 82.0 62.0 1
# 45 heat_secondary_n21 7.96 82.0 82.0 62.0 1
# 46 heat_secondary_n22 7.96 82.0 82.0 62.0 1
# 47 heat_secondary_n23 7.96 82.0 82.0 62.0 1
# 48 heat_secondary_n24 7.96 82.0 82.0 62.0 1
# 49 heat_secondary_n25 7.96 82.0 82.0 62.0 1
# 50 heat_secondary_n26 7.96 82.0 82.0 62.0 1
</HeatNode>

<HeatSource>
@ idx name node supply_node return_node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 primary_source - 1 13 PRESSURE 10.0 0.0 1.0 0.0 3.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name node supply_node return_node mass_flow heat_power run_stat
# 1 primary_explicit_load_1 - 7 19 0.041666666666666664 1.3953333333333333 1
# 2 primary_explicit_load_2 - 8 20 0.041666666666666664 1.3953333333333333 1
# 3 primary_explicit_load_3 - 9 21 0.041666666666666664 1.3953333333333333 1
# 4 primary_explicit_load_4 - 10 22 0.041666666666666664 1.3953333333333333 1
# 5 primary_explicit_load_5 - 11 23 0.041666666666666664 1.3953333333333333 1
# 6 primary_explicit_load_6 - 12 24 0.041666666666666664 1.3953333333333333 1
# 7 secondary_load_1 38 - - 0.076923076923076927 3.2200000000000002 1
# 8 secondary_load_2 39 - - 0.076923076923076927 3.2200000000000002 1
# 9 secondary_load_3 40 - - 0.076923076923076927 3.2200000000000002 1
# 10 secondary_load_4 41 - - 0.076923076923076927 3.2200000000000002 1
# 11 secondary_load_5 42 - - 0.076923076923076927 3.2200000000000002 1
# 12 secondary_load_6 43 - - 0.076923076923076927 3.2200000000000002 1
# 13 secondary_load_7 44 - - 0.076923076923076927 3.2200000000000002 1
# 14 secondary_load_8 45 - - 0.076923076923076927 3.2200000000000002 1
# 15 secondary_load_9 46 - - 0.076923076923076927 3.2200000000000002 1
# 16 secondary_load_10 47 - - 0.076923076923076927 3.2200000000000002 1
# 17 secondary_load_11 48 - - 0.076923076923076927 3.2200000000000002 1
# 18 secondary_load_12 49 - - 0.076923076923076927 3.2200000000000002 1
# 19 secondary_load_13 50 - - 0.076923076923076927 3.2200000000000002 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 heat_pipe_2_4 2 4 10.0 4.9999999999999998e-07 1
# 2 heat_pipe_2_5 2 5 10.0 9.9999999999999995e-07 1
# 3 heat_pipe_3_6 3 6 10.0 1.5e-06 1
# 4 heat_pipe_3_7 3 7 10.0 4.9999999999999998e-07 1
# 5 heat_pipe_4_8 4 8 10.0 9.9999999999999995e-07 1
# 6 heat_pipe_4_9 4 9 10.0 1.5e-06 1
# 7 heat_pipe_5_10 5 10 10.0 4.9999999999999998e-07 1
# 8 heat_pipe_5_11 5 11 10.0 9.9999999999999995e-07 1
# 9 heat_pipe_6_12 6 12 10.0 1.5e-06 1
# 10 heat_pipe_14_13 14 13 10.0 4.9999999999999998e-07 1
# 11 heat_pipe_15_13 15 13 10.0 9.9999999999999995e-07 1
# 12 heat_pipe_16_14 16 14 10.0 1.5e-06 1
# 13 heat_pipe_17_14 17 14 10.0 4.9999999999999998e-07 1
# 14 heat_pipe_18_15 18 15 10.0 9.9999999999999995e-07 1
# 15 heat_pipe_20_16 20 16 10.0 4.9999999999999998e-07 1
# 16 heat_pipe_21_16 21 16 10.0 9.9999999999999995e-07 1
# 17 heat_pipe_22_17 22 17 10.0 1.5e-06 1
# 18 heat_pipe_23_17 23 17 10.0 4.9999999999999998e-07 1
# 19 heat_pipe_24_18 24 18 10.0 9.9999999999999995e-07 1
# 20 heat_pipe_25_26 25 26 10.0 1.5e-06 1
# 21 heat_pipe_25_27 25 27 10.0 4.9999999999999998e-07 1
# 22 heat_pipe_26_28 26 28 10.0 9.9999999999999995e-07 1
# 23 heat_pipe_26_29 26 29 10.0 1.5e-06 1
# 24 heat_pipe_27_30 27 30 10.0 4.9999999999999998e-07 1
# 25 heat_pipe_27_31 27 31 10.0 9.9999999999999995e-07 1
# 26 heat_pipe_28_32 28 32 10.0 1.5e-06 1
# 27 heat_pipe_28_33 28 33 10.0 4.9999999999999998e-07 1
# 28 heat_pipe_29_34 29 34 10.0 9.9999999999999995e-07 1
# 29 heat_pipe_29_35 29 35 10.0 1.5e-06 1
# 30 heat_pipe_30_36 30 36 10.0 4.9999999999999998e-07 1
# 31 heat_pipe_31_38 31 38 10.0 1.5e-06 1
# 32 heat_pipe_31_39 31 39 10.0 4.9999999999999998e-07 1
# 33 heat_pipe_32_40 32 40 10.0 9.9999999999999995e-07 1
# 34 heat_pipe_32_41 32 41 10.0 1.5e-06 1
# 35 heat_pipe_33_42 33 42 10.0 4.9999999999999998e-07 1
# 36 heat_pipe_33_43 33 43 10.0 9.9999999999999995e-07 1
# 37 heat_pipe_34_45 34 45 10.0 4.9999999999999998e-07 1
# 38 heat_pipe_35_46 35 46 10.0 9.9999999999999995e-07 1
# 39 heat_pipe_35_47 35 47 10.0 1.5e-06 1
# 40 heat_pipe_36_48 36 48 10.0 4.9999999999999998e-07 1
# 41 heat_pipe_36_49 36 49 10.0 9.9999999999999995e-07 1
# 42 heat_pipe_37_50 37 50 10.0 1.5e-06 1
</HeatPipe>

<HeatValve>
@ idx name i_node j_node control_type conductance flow_set heat_loss run_stat
# 1 heat_valve_1_3 1 3 OPEN 10.0 0.0 1.5e-06 1
# 2 heat_valve_19_15 19 15 OPEN 10.0 0.0 1.5e-06 1
# 3 heat_valve_30_37 30 37 OPEN 10.0 0.0 9.9999999999999995e-07 1
</HeatValve>

<HeatPump>
@ idx name i_node j_node control_type pressure_gain flow_set heat_loss run_stat
# 1 heat_pump_1_2 1 2 GAIN 0.0 0.0 9.9999999999999995e-07 1
# 2 heat_pump_34_44 34 44 GAIN 0.0 0.0 1.5e-06 1
</HeatPump>

<HeatExchanger>
@ idx name i_node j_node primary_supply_node primary_return_node secondary_return_node secondary_supply_node control_type primary_flow secondary_flow heat_set effectiveness heat_loss run_stat
# 1 three_port_exchanger - 25 12 24 - - EFFECTIVENESS 1.0 1.0 0.0 0.8 0.02 1
</HeatExchanger>
