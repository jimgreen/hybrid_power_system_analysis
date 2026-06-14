<pv_generator>
@ id  name  p_max  p_min  p_fur  rated_power  temp_coefficient  reference_irradiance  reference_temperature
# 1   光伏01  50.0   0      0.0    50.0         -0.004            1000.0                25.0
# 2   光伏02  50.0   0      0.0    50.0         -0.004            1000.0                25.0
# 3   光伏03  40.0   0      0.0    40.0         -0.004            1000.0                25.0
</pv_generator>
<wind_generator>
@ id  name  p_max  p_min  p_fur  rated_power  rated_wind_speed  cut_in_speed  cut_out_speed
# 1   风机01  10     0      0.0    10.0         15.0              5.0           30.0
# 2   风机02  10     0      0.0    10.0         15.0              5.0           30.0
# 3   风机03  10     0      0.0    10.0         15.0              5.0           30.0
# 4   风机04  10     0      0.0    10.0         15.0              5.0           30.0
# 5   风机05  10     0      0.0    10.0         15.0              5.0           30.0
# 6   风机06  10     0      0.0    10.0         15.0              5.0           30.0
# 7   风机07  10     0      0.0    10.0         15.0              5.0           30.0
# 8   风机08  10     0      0.0    10.0         15.0              5.0           30.0
# 9   风机09  10     0      0.0    10.0         15.0              5.0           30.0
# 10  风机10  10     0      0.0    10.0         15.0              5.0           30.0
</wind_generator>
<diesel_generator>
@ id  name  p_max  p_min
# 1   柴发01  300    80
# 2   柴发02  300    80
# 3   柴发03  300    80
# 4   柴发04  300    80
</diesel_generator>
<energyconsumer>
@ id  name  temp_base  temp_factor
# 3   负荷03  5.0        0.0
# 4   负荷04  5.0        0.0
# 5   负荷05  5.0        0.0
# 6   负荷06  5.0        0.0
# 7   负荷07  5.0        0.0
</energyconsumer>
<estorage>
@ id  name  emva  soc_max  soc_min  soc_cur  charge_p_max  dis_charge_p_max
# 1   储能01  50.0  0.9      0.3      0.5      20.0          20.0
# 2   储能02  50.0  0.9      0.3      0.5      20.0          20.0
# 3   储能03  50.0  0.9      0.3      0.5      20.0          20.0
# 4   储能04  50.0  0.9      0.3      0.5      20.0          20.0
# 5   储能05  50.0  0.9      0.3      0.5      20.0          20.0
# 6   储能06  50.0  0.9      0.3      0.5      20.0          20.0
</estorage>
