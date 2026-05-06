<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1.0     0.001   1.0
</PowerBase>

<ACNode>
@ idx name  vbase voltage angle isl run_stat
# 0   bus_1 345   345     0     0   1
# 1   bus_2 345   345     0     0   1
# 2   bus_3 345   345     0     0   1
# 3   bus_4 345   345     0     0   1
# 4   bus_5 345   345     0     0   1
# 5   bus_6 345   345     0     0   1
# 6   bus_7 345   345     0     0   1
# 7   bus_8 345   345     0     0   1
# 8   bus_9 345   345     0     0   1
</ACNode>

<ACBranch>
@ idx name     i_node j_node r      x      b     run_stat i_p i_q i_c j_p j_q j_c
# 0   line_1_4 0      3      0      0.0576 0     1        0   0   0   0   0   0
# 1   line_4_5 3      4      0.017  0.092  0.158 1        0   0   0   0   0   0
# 2   line_5_6 4      5      0.039  0.17   0.358 1        0   0   0   0   0   0
# 3   line_3_6 2      5      0      0.0586 0     1        0   0   0   0   0   0
# 4   line_6_7 5      6      0.0119 0.1008 0.209 1        0   0   0   0   0   0
# 5   line_7_8 6      7      0.0085 0.072  0.149 1        0   0   0   0   0   0
# 6   line_8_2 7      1      0      0.0625 0     1        0   0   0   0   0   0
# 7   line_8_9 7      8      0.032  0.161  0.306 1        0   0   0   0   0   0
# 8   line_9_4 8      3      0.01   0.085  0.176 1        0   0   0   0   0   0
</ACBranch>

<ACLoad>
@ idx name   node pv0 pv1 pv2 qv0 qv1 qv2 run_stat p q current
# 0   load_5 4    90  0   0   30  0   0   1        0 0 0
# 1   load_7 6    100 0   0   35  0   0   1        0 0 0
# 2   load_9 8    125 0   0   50  0   0   1        0 0 0
</ACLoad>

<ACGenerator>
@ idx name    node control_type p_set q_set v_set alpha run_stat p q current
# 0   gen_1_0 0    V            0     0     345   1.0   1        0 0 0
# 1   gen_2_1 1    PV           163   0     345   1.0   1        0 0 0
# 2   gen_3_2 2    PV           85    0     345   1.0   1        0 0 0
</ACGenerator>

<ACShuntCompensator>
@ idx name node control_type q_set g_set b_set v_set run_stat p q current
</ACShuntCompensator>

<ACZeroBranch>
@ idx name i_node j_node run_stat p q current
</ACZeroBranch>

<ACSwitch>
@ idx name i_node j_node status run_stat p q current
</ACSwitch>

<ACTransformer>
@ idx name i_node j_node r x b tap shift run_stat i_p i_q i_c j_p j_q j_c
</ACTransformer>
