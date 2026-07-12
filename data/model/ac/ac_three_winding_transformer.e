<PowerBase>
@ p_base u_unit p_unit i_unit
#     100 kV     MW     kA
</PowerBase>

<ACNode>
@ idx name  vbase voltage angle isl run_stat
#   1 bus_i   110     110     0   0        1
#   2 bus_j   110     110     0   0        1
#   3 bus_k   110     110     0   0        1
</ACNode>

<ACGenerator>
@ idx name    node control_type p_set q_set v_set alpha run_stat
#   1 slack_i    1 V                 0     0   110     1        1
</ACGenerator>

<ACLoad>
@ idx name   node pbase pv0  pv1 pv2 qbase qv0  qv1 qv2 run_stat
#   1 load_j    2   100 0.12    0   0   100 0.04    0   0        1
#   2 load_k    3   100 0.08    0   0   100 0.03    0   0        1
</ACLoad>

<ACThreeWindingTransformer>
@ idx name     i_node j_node k_node i_r   i_x  j_r   j_x  k_r   k_x  gt     bt     i_tap i_shift j_tap j_shift k_tap k_shift run_stat
#   1 tr3_main      1      2      3 0.002 0.04 0.003 0.05 0.004 0.06 0.0001 -0.001   1.02     2.0   1.0     0.0  0.98    -1.0        1
</ACThreeWindingTransformer>
