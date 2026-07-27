<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV kW kA
</PowerBase>

<DCNode>
@ idx name vbase voltage isl run_stat
#   1 nd_1   100     100   0        1
#   2 nd_2   100     100   0        1
#   3 nd_3   100     100   0        1
#   4 nd_4   100     100   0        1
#   5 nd_5   100     100   0        1
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
#   1 line_1_0_3      1      4 0.01        1
#   2 line_1_0_3      1      4 0.01        1
#   3 line_1_0_3      1      4 0.01        1
#   4 line_1_0_3      1      4 0.01        1
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat
#   1 load_1_1    1   1.0 100   0   0        1
#   2 load_1_2    2   1.0   0   0  50        1
#   3 load_1_3    3   1.0  50  80   0        1
#   4 load_1_3    4   1.0  70  20  80        1
</DCLoad>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
#   1 gen_1_1    5            V   110     0     0        1
#   2 gen_2_1    4            P   110    50     0        1
#   3 gen_3_1    4            I   110    50 0.001        1
</DCGenerator>

<DCZeroBranch>
@ idx name i_node j_node run_stat
#   1 zbr_1_2_3      2      4        1
#   2 zbr_1_2_3      2      4        1
#   3 zbr_1_2_3      2      4        1
#   4 zbr_1_2_3      2      4        1
</DCZeroBranch>

<DCSwitch>
@ idx name i_node j_node status run_stat
#   1 sw_1_2_3      3      4      1        1
</DCSwitch>

<DCBreak>
@ idx name i_node j_node status run_stat
#   1 sw_1_2_3      3      4      1        1
#   2 sw_1_2_3      3      4      1        1
</DCBreak>

<DCDCConverter>
@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat
# 1 conv_1_2_3 5 4 0.005 0.005 V NONE 0 0 110 1
</DCDCConverter>
