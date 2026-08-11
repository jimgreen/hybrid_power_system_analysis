<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure temperature supply_temperature return_temperature run_stat
# 1 primary_source_supply 10.0 90.0 90.0 78.0 1
# 2 primary_hx_supply 9.0 89.0 89.0 78.0 1
# 3 primary_hx_return 5.0 78.0 89.0 78.0 1
# 4 primary_source_return 4.0 77.0 89.0 77.0 1
# 5 secondary_hx_node 8.0 87.0 87.0 75.0 1
# 6 secondary_load_node 7.0 86.0 86.0 74.0 1
</HeatNode>

<HeatSource>
@ idx name supply_node return_node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 primary_source 1 4 PRESSURE 10.0 0.0 1.0 0.0 2.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name node mass_flow heat_power run_stat
# 1 secondary_load 6 1.0 50.0 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 primary_supply_pipe 1 2 1.0 0.0 1
# 2 primary_return_pipe 3 4 1.0 0.0 1
# 3 secondary_supply_pipe 5 6 1.0 0.0 1
</HeatPipe>

<HeatExchanger>
@ idx name primary_supply_node primary_return_node j_node control_type primary_flow secondary_flow heat_set effectiveness heat_loss run_stat
# 1 three_port_heat_exchanger 2 3 5 EFFECTIVENESS 1.0 1.0 0.0 0.8 0.0 1
</HeatExchanger>
