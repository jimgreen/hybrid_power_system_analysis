<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure supply_temperature return_temperature run_stat
# 1 heat_n1 10.00 90.0 50.0 1
# 2 heat_n2 6.00 88.0 50.0 1
# 3 heat_n3 8.00 87.0 49.0 1
# 4 heat_n4 5.75 85.0 48.0 1
# 5 heat_n5 5.25 83.0 47.0 1
</HeatNode>

<HeatSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 heat_source 1 PRESSURE 10.0 0.0 1.0 0.0 5.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name node mass_flow heat_power run_stat
# 1 heat_load_2 2 0.5 30.0 1
# 2 heat_load_4 4 0.8 45.0 1
# 3 heat_load_5 5 0.7 35.0 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 heat_pipe_12 1 2 1.0 0.08 1
# 2 heat_pipe_45 4 5 1.0 0.05 1
</HeatPipe>

<HeatValve>
@ idx name i_node j_node control_type conductance flow_set heat_loss run_stat
# 1 heat_valve_34 3 4 OPEN 1.0 0.0 0.03 1
</HeatValve>

<HeatPump>
@ idx name i_node j_node control_type pressure_gain flow_set heat_loss run_stat
# 1 heat_pump_23 2 3 GAIN 2.0 1.5 0.02 1
</HeatPump>
