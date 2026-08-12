<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure temperature run_stat
# 1 heat_source_supply 10.0 90.0 1
# 2 heat_load_supply 9.0 88.0 1
# 3 heat_load_return 5.0 70.0 1
# 4 heat_source_return 4.0 68.0 1
</HeatNode>

<HeatSource>
@ idx name supply_node return_node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature_set return_temperature_set run_stat
# 1 heat_source 1 4 PRESSURE 10.0 0.0 1.0 0.0 3.0 90.0 50.0 1
</HeatSource>

<HeatStorage>
@ idx name supply_node return_node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature_set return_temperature_set run_stat
# 1 heat_storage_discharge 1 4 FLOW 10.0 0.2 1.0 -0.5 0.5 85.0 55.0 1
# 2 heat_storage_charge 1 4 FLOW 10.0 -0.1 1.0 -0.5 0.5 80.0 60.0 1
</HeatStorage>

<HeatLoad>
@ idx name supply_node return_node mass_flow heat_power run_stat
# 1 heat_load 2 3 1.0 50.0 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 heat_supply_pipe 1 2 1.0 0.0 1
# 2 heat_return_pipe 3 4 1.0 0.0 1
</HeatPipe>
