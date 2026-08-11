<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure temperature supply_temperature return_temperature run_stat
# 1 heat_supply_n1 10 90.0 90.0 70.0 1
# 2 heat_supply_n2 9.99 90.0 90.0 70.0 1
# 3 heat_return_n1 5 70.0 90.0 70.0 1
# 4 heat_return_n2 5.01 70.0 90.0 70.0 1
# 5 heat_secondary_n1 8 82.0 82.0 62.0 1
# 6 heat_secondary_n2 7.99 82.0 82.0 62.0 1
# 7 heat_secondary_n3 7.99 82.0 82.0 62.0 1
# 8 heat_secondary_n4 7.98 82.0 82.0 62.0 1
# 9 heat_secondary_n5 7.98 82.0 82.0 62.0 1
# 10 heat_secondary_n6 7.98 82.0 82.0 62.0 1
</HeatNode>

<HeatSource>
@ idx name node supply_node return_node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 primary_source - 1 3 PRESSURE 10.0 0.0 1.0 0.0 3.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name node supply_node return_node mass_flow heat_power run_stat
# 1 primary_explicit_load_1 - 2 4 0.25 8.3719999999999999 1
# 2 secondary_load_1 8 - - 0.33333333333333331 13.953333333333333 1
# 3 secondary_load_2 9 - - 0.33333333333333331 13.953333333333333 1
# 4 secondary_load_3 10 - - 0.33333333333333331 13.953333333333333 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 heat_pipe_5_6 5 6 10.0 4.9999999999999998e-07 1
# 2 heat_pipe_5_7 5 7 10.0 9.9999999999999995e-07 1
# 3 heat_pipe_6_8 6 8 10.0 1.5e-06 1
# 4 heat_pipe_6_9 6 9 10.0 4.9999999999999998e-07 1
# 5 heat_pipe_7_10 7 10 10.0 9.9999999999999995e-07 1
</HeatPipe>

<HeatValve>
@ idx name i_node j_node control_type conductance flow_set heat_loss run_stat
# 1 heat_valve_4_3 4 3 OPEN 10.0 0.0 1.5e-06 1
</HeatValve>

<HeatPump>
@ idx name i_node j_node control_type pressure_gain flow_set heat_loss run_stat
# 1 heat_pump_1_2 1 2 GAIN 0.0 0.0 9.9999999999999995e-07 1
</HeatPump>

<HeatExchanger>
@ idx name i_node j_node primary_supply_node primary_return_node secondary_return_node secondary_supply_node control_type primary_flow secondary_flow heat_set effectiveness heat_loss run_stat
# 1 three_port_exchanger - 5 2 4 - - EFFECTIVENESS 1.0 1.0 0.0 0.8 0.02 1
</HeatExchanger>
