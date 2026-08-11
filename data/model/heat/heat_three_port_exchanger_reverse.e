<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure temperature supply_temperature return_temperature run_stat
# 1 primary_hx_node 10.0 90.0 90.0 78.0 1
# 2 secondary_hx_return 2.0 75.0 75.0 75.0 1
# 3 secondary_hx_supply 8.0 87.0 87.0 87.0 1
# 4 secondary_load_supply 7.0 86.0 86.0 86.0 1
# 5 secondary_load_return 3.0 75.0 75.0 75.0 1
</HeatNode>

<HeatSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 primary_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name supply_node return_node mass_flow heat_power run_stat
# 1 secondary_load 4 5 1.0 50.0 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 secondary_supply_pipe 3 4 1.0 0.0 1
# 2 secondary_return_pipe 5 2 1.0 0.0 1
</HeatPipe>

<HeatExchanger>
@ idx name i_node secondary_return_node secondary_supply_node control_type primary_flow secondary_flow heat_set effectiveness heat_loss run_stat
# 1 reverse_three_port_heat_exchanger 1 2 3 EFFECTIVENESS 1.0 1.0 0.0 0.8 0.0 1
</HeatExchanger>
