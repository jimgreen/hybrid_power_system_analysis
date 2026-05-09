<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1.0     1.0     1.0
</PowerBase>

<ACNode>
@ idx name vbase voltage angle isl run_stat
# 0   nd_0 1.0   1       0     0   1
# 1   nd_1 1.0   1       0     0   1
# 2   nd_2 1.0   1       0     0   1
# 3   nd_3 1.0   1       0     0   1
# 4   nd_4 1.0   1       0     0   1
# 5   nd_5 1.0   1       0     0   1
# 6   nd_6 1.0   1       0     0   1
# 7   nd_7 1.0   1       0     0   1
# 8   nd_8 1.0   1       0     0   1
# 9   nd_9 1.0   1       0     0   1
</ACNode>

<ACBranch>
@ idx name i_node j_node r x b run_stat
# 0 line_0_1 0 1 0.01 0.05 0.02 1
# 1 line_0_2 0 2 0.015 0.06 0.025 1
# 2 line_1_3 1 3 0.02 0.08 0.03 1
# 3 line_2_4 2 4 0.018 0.07 0.028 1
# 4 line_3_5 3 5 0.025 0.09 0.035 1
# 5 line_4_6 4 6 0.022 0.085 0.032 1
# 6 line_5_7 5 7 0.028 0.095 0.04 1
# 7 line_6_8 6 8 0.024 0.09 0.036 1
# 8 line_7_9 7 9 0.026 0.1 0.038 1
# 9 line_8_9 8 9 0.015 0.055 0.022 1
# 10 line_1_2 1 2 0.012 0.045 0.018 1
# 11 line_3_4 3 4 0.02 0.07 0.03 1
# 12 line_5_6 5 6 0.018 0.065 0.026 1
# 13 line_7_8 7 8 0.022 0.08 0.034 1
</ACBranch>

<ACLoad>
@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat
# 0 load_3 3 1.0 50 30 20 1.0 40 40 20 1
# 1 load_4 4 1.0 60 20 20 1.0 50 30 20 1
# 2 load_5 5 1.0 40 40 20 1.0 30 40 30 1
# 3 load_6 6 1.0 50 30 20 1.0 50 30 20 1
# 4 load_7 7 1.0 70 20 10 1.0 60 20 20 1
# 5 load_8 8 1.0 50 30 20 1.0 40 30 30 1
# 6 load_9 9 1.0 60 20 20 1.0 50 30 20 1
</ACLoad>

<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha run_stat
# 0 gen_v0 0 V 0 0 1.06 1.0 1
# 1 gen_pv1 1 PV 100 0 1.02 1.0 1
# 2 gen_pv2 2 PV 120 0 1.01 1.0 1
# 3 gen_pq9 9 PQ 50 10 0 1.0 1
</ACGenerator>

<ACShuntCompensator>
@ idx name node control_type q_set g_set b_set v_set run_stat
# 0 shunt_3 3 Q 20 0.0 0.0 0 1
# 1 shunt_4 4 Z 0 0.0 -0.1 0 1
# 2 shunt_5 5 V 0 0.0 0.0 1 1
</ACShuntCompensator>

<ACZeroBranch>
@ idx name i_node j_node run_stat
# 0 zbr_5_6 5 6 1
</ACZeroBranch>

<ACSwitch>
@ idx name i_node j_node status run_stat
</ACSwitch>

<ACBreak>
@ idx name i_node j_node status run_stat
# 0 sw_7_8 7 8 1 1
</ACBreak>

<ACTransformer>
@ idx name i_node j_node r x gt bt tap shift run_stat
# 0 tf_2_5 2 5 0.01 0.1 0.0 0 1.05 0.05 1
</ACTransformer>
