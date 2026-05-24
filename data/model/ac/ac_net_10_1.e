<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV kW kA
</PowerBase>

<ACNode>
@ idx name vbase voltage angle isl run_stat
#   1 nd_0   1.0       1     0   0        1
#   2 nd_1   1.0       1     0   0        1
#   3 nd_2   1.0       1     0   0        1
#   4 nd_3   1.0       1     0   0        1
#   5 nd_4   1.0       1     0   0        1
#   6 nd_5   1.0       1     0   0        1
#   7 nd_6   1.0       1     0   0        1
#   8 nd_7   1.0       1     0   0        1
#   9 nd_8   1.0       1     0   0        1
#  10 nd_9   1.0       1     0   0        1
</ACNode>

<ACBranch>
@ idx name i_node j_node r x b run_stat
#   1 line_0_1      1      2  0.01  0.05  0.02        1
#   2 line_0_2      1      3 0.015  0.06 0.025        1
#   3 line_1_3      2      4  0.02  0.08  0.03        1
#   4 line_2_4      3      5 0.018  0.07 0.028        1
#   5 line_3_5      4      6 0.025  0.09 0.035        1
#   6 line_4_6      5      7 0.022 0.085 0.032        1
#   7 line_5_7      6      8 0.028 0.095  0.04        1
#   8 line_6_8      7      9 0.024  0.09 0.036        1
#   9 line_7_9      8     10 0.026   0.1 0.038        1
#  10 line_8_9      9     10 0.015 0.055 0.022        1
#  11 line_1_2      2      3 0.012 0.045 0.018        1
#  12 line_3_4      4      5  0.02  0.07  0.03        1
#  13 line_5_6      6      7 0.018 0.065 0.026        1
#  14 line_7_8      8      9 0.022  0.08 0.034        1
</ACBranch>

<ACLoad>
@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat
#   1 load_3    4   1.0   0   0   0   1.0   0   0   0        1
#   2 load_4    5   1.0   0   0   0   1.0   0   0   0        1
#   3 load_5    6   1.0   0   0   0   1.0   0   0   0        1
#   4 load_6    7   1.0   0   0   0   1.0   0   0   0        1
#   5 load_7    8   1.0   0   0   0   1.0   0   0   0        1
#   6 load_8    9   1.0   0   0   0   1.0   0   0   0        1
#   7 load_9   10   1.0   0   0   0   1.0   0   0   0        1
</ACLoad>

<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha run_stat
#   1 gen_v0     1            V     0     0  1.06   1.0        1
#   2 gen_pv1    2           PV   100     0  1.02   1.0        1
#   3 gen_pv2    3           PV   120     0  1.01   1.0        1
#   4 gen_pq9   10           PQ    50    10     0   1.0        1
</ACGenerator>

<ACShuntCompensator>
@ idx name node control_type q_set g_set b_set v_set run_stat
#   1 shunt_3    4            Q    20   0.0   0.0     0        1
#   2 shunt_4    5            Z     0   0.0  -0.1     0        1
#   3 shunt_5    6            V     0   0.0   0.0     1        1
</ACShuntCompensator>

<ACZeroBranch>
@ idx name i_node j_node run_stat
#   1 zbr_5_6      6      7        1
</ACZeroBranch>

<ACSwitch>
@ idx name i_node j_node status run_stat
</ACSwitch>

<ACBreak>
@ idx name i_node j_node status run_stat
#   1 sw_7_8      8      9      1        1
</ACBreak>

<ACTransformer>
@ idx name i_node j_node r x gt bt tap shift run_stat
#   1 tf_2_5      3      6 0.01 0.1 0.0  0 1.05  0.05        1
</ACTransformer>
