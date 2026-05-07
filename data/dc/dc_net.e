<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1.0     1.0     1.0
</PowerBase>

<DCNode>
@ idx name vbase voltage isl run_stat
# 0   nd_1 100   100     0   1
# 1   nd_2 100   100     0   1
# 2   nd_3 100   100     0   1
# 3   nd_4 100   100     0   1
# 4   nd_5 100   100     0   1
</DCNode>

<DCBranch>
@ idx name       i_node j_node r    run_stat i_p j_p current
# 0   line_1_0_3 0      3      0.01 1        0   0   0
# 1   line_1_0_3 0      3      0.01 1        0   0   0
# 2   line_1_0_3 0      3      0.01 1        0   0   0
# 3   line_1_0_3 0      3      0.01 1        0   0   0
</DCBranch>

<DCLoad>
@ idx name     node pbase pv0 pv1 pv2 run_stat p current
# 0   load_1_1 0    1.0    100 0   0   1        0 0
# 1   load_1_2 1    1.0    0   0   50  1        0 0
# 2   load_1_3 2    1.0    50  80  0   1        0 0
# 3   load_1_3 3    1.0    70  20  80  1        0 0
</DCLoad>

<DCGenerator>
@ idx name    node control_type v_set p_set i_set run_stat p current
# 0   gen_1_1 4    V            110   0     0     1        0 0
# 1   gen_2_1 3    P            110   50    0     1        0 0
# 2   gen_3_1 3    I            110   50    0.001 1        0 0
</DCGenerator>

<DCZeroBranch>
@ idx name      i_node j_node run_stat p current
# 0   zbr_1_2_3 1      3      1        0 0
# 1   zbr_1_2_3 1      3      1        0 0
# 2   zbr_1_2_3 1      3      1        0 0
# 3   zbr_1_2_3 1      3      1        0 0
</DCZeroBranch>

<DCSwitch>
@ idx name     i_node j_node status run_stat p current
# 1   sw_1_2_3 2      3      1      1        0 0
</DCSwitch>

<DCBreak>
@ idx name     i_node j_node status run_stat p current
# 0   sw_1_2_3 2      3      1      1        0 0
# 2   sw_1_2_3 2      3      1      1        0 0
</DCBreak>

<DCDCConverter>
@ idx name       i_node j_node r1    r2    control_type p_set i_set v_set run_stat i_p j_p i_c j_c
# 0   conv_1_2_3 4      3      0.005 0.005 V            0     0     110   1        0   0   0   0
</DCDCConverter>
