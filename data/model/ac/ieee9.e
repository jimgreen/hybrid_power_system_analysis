<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV MW kA
</PowerBase>

<ACNode>
@ idx name  vbase voltage angle isl run_stat
#   1 bus_1   345     345     0   0        1
#   2 bus_2   345     345     0   0        1
#   3 bus_3   345     345     0   0        1
#   4 bus_4   345     345     0   0        1
#   5 bus_5   345     345     0   0        1
#   6 bus_6   345     345     0   0        1
#   7 bus_7   345     345     0   0        1
#   8 bus_8   345     345     0   0        1
#   9 bus_9   345     345     0   0        1
</ACNode>

<ACBranch>
@ idx name i_node j_node r x b run_stat
#   1 line_1_4      1      4      0 0.0576     0        1
#   2 line_4_5      4      5  0.017  0.092 0.158        1
#   3 line_5_6      5      6  0.039   0.17 0.358        1
#   4 line_3_6      3      6      0 0.0586     0        1
#   5 line_6_7      6      7 0.0119 0.1008 0.209        1
#   6 line_7_8      7      8 0.0085  0.072 0.149        1
#   7 line_8_2      8      2      0 0.0625     0        1
#   8 line_8_9      8      9  0.032  0.161 0.306        1
#   9 line_9_4      9      4   0.01  0.085 0.176        1
</ACBranch>

<ACLoad>
@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat
#   1 load_5    5   1.0  90 0.0 0.0   1.0  30 0.0 0.0        1
#   2 load_7    7   1.0 100 0.0 0.0   1.0  35 0.0 0.0        1
#   3 load_9    9   1.0 125 0.0 0.0   1.0  50 0.0 0.0        1
</ACLoad>

<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha run_stat
#   1 gen_1_0    1            V     0     0   345   1.0        1
#   2 gen_2_1    2           PV   163     0   345   1.0        1
#   3 gen_3_2    3           PV    85     0   345   1.0        1
</ACGenerator>

<ACShuntCompensator>
@ idx name node control_type q_set g_set b_set v_set run_stat
</ACShuntCompensator>

<ACZeroBranch>
@ idx name i_node j_node run_stat
</ACZeroBranch>

<ACSwitch>
@ idx name i_node j_node status run_stat
</ACSwitch>

<ACTransformer>
@ idx name i_node j_node r x gt bt tap shift run_stat
</ACTransformer>
