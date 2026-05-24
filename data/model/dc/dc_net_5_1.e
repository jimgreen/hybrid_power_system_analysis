<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV kW kA
</PowerBase>

<DCNode>
@ idx name  vbase voltage isl run_stat
#   1 nd_1  100.0     100   0        1
#   2 nd_2  100.0     100   0        1
#   3 nd_3  100.0     100   0        1
#   4 nd_21 100.0     100   0        1
#   5 nd_22 100.0     100   0        1
#   6 nd_23 100.0     100   0        1
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
#   1 line_0_1        1      2   0.01        1
#   2 line_1_2        2      3   0.02        1
#   3 line_22_23      4      5 0.0003        1
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat
#   1 ld_1    1   1.0 100   0   0        1
</DCLoad>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
#   1 gen_v1    3            V   120   100     0        1
#   2 gen_i4    6            V   130     0 0.001        1
</DCGenerator>

<DCZeroBranch>
@ idx name i_node j_node run_stat
</DCZeroBranch>

<DCSwitch>
@ idx name i_node j_node status run_stat
</DCSwitch>

<DCDCConverter>
@ idx name i_node j_node r1 r2 control_type p_set i_set v_set run_stat
#   1 conv_link      4      3 0.015 0.015            V     0      0   100        1
#   2 conv_7         5      6     0     0            P   100 0.0005     0        1
</DCDCConverter>
