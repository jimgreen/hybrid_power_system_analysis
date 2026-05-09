<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1.0     1.0     1.0
</PowerBase>

<DCNode>
@ idx name  vbase voltage isl run_stat
# 0   nd_1  100.0 100     0   1
# 1   nd_2  100.0 100     0   1
# 2   nd_3  100.0 100     0   1
# 3   nd_21 100.0 100     0   1
# 4   nd_22 100.0 100     0   1
# 5   nd_23 100.0 100     0   1
</DCNode>

<DCBranch>
@ idx name       i_node j_node r      run_stat i_p j_p current
# 0   line_0_1   0      1      0.01   1        0   0   0
# 1   line_1_2   1      2      0.02   1        0   0   0
# 29  line_22_23 3      4      0.0003 1        0   0   0
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat p current
# 0   ld_1 0    1.0    100 0   0   1        0 0
</DCLoad>

<DCGenerator>
@ idx name   node control_type v_set p_set i_set run_stat p current
# 0   gen_v1 2    V            120   100   0     1        0 0
# 10  gen_i4 5    V            130   0     0.001 1        0 0
</DCGenerator>

<DCZeroBranch>
@ idx name i_node j_node run_stat p current
</DCZeroBranch>

<DCSwitch>
@ idx name i_node j_node status run_stat p current
</DCSwitch>

<DCDCConverter>
@ idx name      i_node j_node r1    r2    control_type p_set i_set  v_set run_stat i_p j_p i_c j_c
# 6   conv_link 3      2      0.015 0.015 V            0     0      100   1        0   0   0   0
# 7   conv_7    4      5      0     0     P            100   0.0005 0     1        0   0   0   0
</DCDCConverter>
