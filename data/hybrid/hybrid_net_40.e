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
@ idx name     i_node j_node r     x     b     run_stat i_p i_q i_c j_p j_q j_c
# 0   line_0_1 0      1      0.01  0.05  0.02  1        0   0   0   0   0   0
# 1   line_0_2 0      2      0.015 0.06  0.025 1        0   0   0   0   0   0
# 2   line_1_3 1      3      0.02  0.08  0.03  1        0   0   0   0   0   0
# 3   line_2_4 2      4      0.018 0.07  0.028 1        0   0   0   0   0   0
# 4   line_3_5 3      5      0.025 0.09  0.035 1        0   0   0   0   0   0
# 5   line_4_6 4      6      0.022 0.085 0.032 1        0   0   0   0   0   0
# 6   line_5_7 5      7      0.028 0.095 0.04  1        0   0   0   0   0   0
# 7   line_6_8 6      8      0.024 0.09  0.036 1        0   0   0   0   0   0
# 8   line_7_9 7      9      0.026 0.1   0.038 1        0   0   0   0   0   0
# 9   line_8_9 8      9      0.015 0.055 0.022 1        0   0   0   0   0   0
# 10  line_1_2 1      2      0.012 0.045 0.018 1        0   0   0   0   0   0
# 11  line_3_4 3      4      0.02  0.07  0.03  1        0   0   0   0   0   0
# 12  line_5_6 5      6      0.018 0.065 0.026 1        0   0   0   0   0   0
# 13  line_7_8 7      8      0.022 0.08  0.034 1        0   0   0   0   0   0
</ACBranch>

<ACLoad>
@ idx name   node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat p q current
# 0   load_3 3    1.0    50  30  20  1.0  40  40  20  1        0 0 0
# 1   load_4 4    1.0    60  20  20  1.0  50  30  20  1        0 0 0
# 2   load_5 5    1.0    40  40  20  1.0  30  40  30  1        0 0 0
# 3   load_6 6    1.0    50  30  20  1.0  50  30  20  1        0 0 0
# 4   load_7 7    1.0    70  20  10  1.0  60  20  20  1        0 0 0
# 5   load_8 8    1.0    50  30  20  1.0  40  30  30  1        0 0 0
# 6   load_9 9    1.0    60  20  20  1.0  50  30  20  1        0 0 0
</ACLoad>

<ACGenerator>
@ idx name    node control_type p_set q_set v_set alpha run_stat p q current
# 0   gen_v0  0    V            0     0     1.06  1.0   1        0 0 0
# 1   gen_pv1 1    PV           100   0     1.02  1.0   1        0 0 0
# 2   gen_pv2 2    PV           120   0     1.01  1.0   1        0 0 0
# 3   gen_pq9 9    PQ           50    10    0     1.0   1        0 0 0
</ACGenerator>

<ACShuntCompensator>
@ idx name    node control_type q_set g_set b_set v_set run_stat p q current
# 0   shunt_3 3    Q            20    0.0   0.0   0     1        0 0 0
# 1   shunt_4 4    Z            0     0.0   -0.1  0     1        0 0 0
# 2   shunt_5 5    V            0     0.0   0.0   1     1        0 0 0
</ACShuntCompensator>

<ACZeroBranch>
@ idx name    i_node j_node run_stat p q current
# 0   zbr_5_6 5      6      1        0 0 0
</ACZeroBranch>

<ACSwitch>
@ idx name   i_node j_node status run_stat p q current
</ACSwitch>

<ACBreak>
@ idx name   i_node j_node status run_stat p q current
# 0   sw_7_8 7      8      1      1        0 0 0
</ACBreak>

<ACTransformer>
@ idx name   i_node j_node r    x   b   tap  shift run_stat i_p i_q i_c j_p j_q j_c
# 0   tf_2_5 2      5      0.01 0.1 0.0 1.05 0.05  1        0   0   0   0   0   0
</ACTransformer>

<DCNode>
@ idx name  vbase voltage isl run_stat
# 0   nd_1  100.0 100     0   1
# 1   nd_2  100.0 100     0   1
# 2   nd_3  100.0 100     0   1
# 3   nd_4  100.0 100     0   1
# 4   nd_5  100.0 100     0   1
# 5   nd_6  100.0 100     0   1
# 6   nd_7  100.0 100     0   1
# 7   nd_8  100.0 100     0   1
# 8   nd_9  100.0 100     0   1
# 9   nd_10 100.0 100     0   1
# 10  nd_11 100.0 100     0   1
# 11  nd_12 100.0 100     0   1
# 12  nd_13 100.0 100     0   1
# 13  nd_14 100.0 100     0   1
# 14  nd_15 100.0 100     0   1
# 15  nd_16 100.0 100     0   1
# 16  nd_17 100.0 100     0   1
# 17  nd_18 100.0 100     0   1
# 18  nd_19 100.0 100     0   1
# 19  nd_20 100.0 100     0   1
# 20  nd_21 100.0 100     0   1
# 21  nd_22 100.0 100     0   1
# 22  nd_23 100.0 100     0   1
# 23  nd_24 100.0 100     0   1
# 24  nd_25 100.0 100     0   1
# 25  nd_26 100.0 100     0   1
# 26  nd_27 100.0 100     0   1
# 27  nd_28 100.0 100     0   1
# 28  nd_29 100.0 100     0   1
# 29  nd_30 100.0 100     0   1
</DCNode>

<DCBranch>
@ idx name       i_node j_node r    run_stat i_p j_p current
# 0   line_0_1   0      1      0.1  1        0   0   0
# 1   line_1_2   1      2      0.02 1        0   0   0
# 2   line_2_3   2      3      0.03 1        0   0   0
# 3   line_3_4   3      4      0.1  1        0   0   0
# 4   line_4_0   4      0      0.02 1        0   0   0
# 5   line_5_6   5      6      0.1  1        0   0   0
# 6   line_6_7   6      7      0.02 1        0   0   0
# 7   line_7_8   7      8      0.03 1        0   0   0
# 8   line_8_9   8      9      0.1  1        0   0   0
# 9   line_9_5   9      5      0.02 1        0   0   0
# 10  line_10_11 10     11     0.1  1        0   0   0
# 11  line_11_12 11     12     0.02 1        0   0   0
# 12  line_12_13 12     13     0.03 1        0   0   0
# 13  line_13_14 13     14     0.1  1        0   0   0
# 14  line_14_10 14     10     0.02 1        0   0   0
# 15  line_15_16 15     16     0.1  1        0   0   0
# 16  line_16_17 16     17     0.02 1        0   0   0
# 17  line_17_18 17     18     0.03 1        0   0   0
# 18  line_18_19 18     19     0.1  1        0   0   0
# 19  line_19_15 19     15     0.02 1        0   0   0
# 20  line_4_5   4      5      0.04 1        0   0   0
# 21  line_9_10  9      10     0.05 1        0   0   0
# 22  line_14_15 14     15     0.03 1        0   0   0
# 23  line_2_6   2      6      0.06 1        0   0   0
# 24  line_8_12  8      12     0.07 1        0   0   0
# 25  line_13_17 13     17     0.05 1        0   0   0
# 26  line_0_19  0      19     0.08 1        0   0   0
# 27  line_20_21 20     21     0.02 1        0   0   0
# 28  line_21_22 21     22     0.03 1        0   0   0
# 29  line_22_23 22     23     0.1  1        0   0   0
# 30  line_23_24 23     24     0.2  1        0   0   0
# 31  line_24_20 24     20     0.04 1        0   0   0
# 32  line_25_26 25     26     0.1  1        0   0   0
# 33  line_26_27 26     27     0.2  1        0   0   0
# 34  line_27_28 27     28     0.03 1        0   0   0
# 35  line_28_29 28     29     0.1  1        0   0   0
# 36  line_29_25 29     25     0.2  1        0   0   0
</DCBranch>

<DCLoad>
@ idx name    node pbase pv0 pv1 pv2 run_stat p current
# 0   load_1  0    1.0    100 10  10  1        0 0
# 1   load_2  2    1.0    100 20  0   1        0 0
# 2   load_3  4    1.0    100 10  20  1        0 0
# 3   load_4  6    1.0    100 20  0   1        0 0
# 4   load_5  8    1.0    100 10  10  1        0 0
# 5   load_6  10   1.0   100 20  0   1        0 0
# 6   load_7  12   1.0   100 10  20  1        0 0
# 7   load_8  14   1.0   100 20  0   1        0 0
# 8   load_9  16   1.0   100 10  10  1        0 0
# 9   load_10 18   1.0   100 20  0   1        0 0
# 10  load_11 19   1.0   100 20  20  1        0 0
# 11  load_12 20   1.0   100 10  0   1        0 0
# 12  load_13 22   1.0   100 20  10  1        0 0
# 13  load_14 24   1.0   100 10  0   1        0 0
# 14  load_15 26   1.0   100 20  20  1        0 0
# 15  load_16 28   1.0   100 10  0   1        0 0
</DCLoad>

<DCGenerator>
@ idx name   node control_type v_set p_set i_set  run_stat p current
# 0   gen_v1 3    V            160   100   0      1        0 0
# 1   gen_v2 10   P            100   100   0      1        0 0
# 2   gen_v3 17   P            100   150   0      1        0 0
# 3   gen_p1 5    P            110   200   0      1        0 0
# 4   gen_i1 7    I            110   0     0.0015 1        0 0
# 5   gen_p2 11   P            110   180   0      1        0 0
# 6   gen_i2 13   I            110   0     0.0012 1        0 0
# 7   gen_p3 15   P            110   220   0      1        0 0
# 8   gen_i3 19   I            110   0     0.001  1        0 0
# 9   gen_p4 21   P            0     150   0      1        0 0
# 10  gen_i4 23   I            0     0     0.001  1        0 0
# 11  gen_p5 25   V            140   200   0      1        0 0
# 12  gen_i5 27   I            0     0     0.0012 1        0 0
# 13  gen_p6 29   P            0     120   0      1        0 0
</DCGenerator>

<DCZeroBranch>
@ idx name      i_node j_node run_stat p current
# 0   zbr_1_2   1      2      1        0 0
# 1   zbr_3_4   3      4      1        0 0
# 2   zbr_6_7   6      7      1        0 0
# 3   zbr_9_10  9      10     1        0 0
# 4   zbr_11_12 11     12     1        0 0
# 5   zbr_14_15 14     15     1        0 0
# 6   zbr_16_17 16     17     1        0 0
# 7   zbr_20_21 20     21     1        0 0
</DCZeroBranch>

<DCSwitch>
@ idx name     i_node j_node status run_stat p current
# 1   sw_2_3   2      3      1      1        0 0
# 3   sw_6_8   6      8      1      1        0 0
# 5   sw_10_12 10     12     1      1        0 0
# 7   sw_15_17 15     17     1      1        0 0
</DCSwitch>

<DCBreak>
@ idx name     i_node j_node status run_stat p current
# 0   sw_0_1   0      1      1      1        0 0
# 2   sw_4_5   4      5      1      1        0 0
# 4   sw_9_11  9      11     1      1        0 0
# 6   sw_13_14 13     14     1      1        0 0
# 8   sw_18_19 18     19     1      1        0 0
</DCBreak>

<DCDCConverter>
@ idx name      i_node j_node r1    r2    control_type p_set i_set  v_set run_stat i_p j_p i_c j_c
# 0   conv_1    4      5      0.05  0.05  P            200   0      0     1        0   0   0   0
# 1   conv_2    8      9      0.1   0.1   P            100   0      0     1        0   0   0   0
# 2   conv_3    12     13     0.075 0.075 I            0     0.0015 0     1        0   0   0   0
# 3   conv_4    16     17     0.06  0.06  P            150   0      0     1        0   0   0   0
# 4   conv_5    18     19     0.09  0.09  I            0     0.001  0     1        0   0   0   0
# 5   conv_6    0      15     0.125 0.125 P            100   0      0     1        0   0   0   0
# 6   conv_link 20     10     0.075 0.075 V            0     0      120   1        0   0   0   0
# 7   conv_7    22     23     0.05  0.05  P            120   0      0     1        0   0   0   0
# 8   conv_8    26     27     0.06  0.06  I            0     0.0008 0     1        0   0   0   0
</DCDCConverter>

<DCACConverter>
@ idx name    ac_node dc_node r1   r2   control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat dc_p ac_p ac_q dc_i ac_i
# 0   inv_dcv 7       22      0.02 0.02 DCV          0        0        0        117.2556 1        0    0    0    0    0
# 1   inv_acv 8       24      0.02 0.02 ACV          0        0        0.823466 0        1        0    0    0    0    0
# 2   inv_acp 9       28      0.02 0.02 ACP          -10      0        0        0        1        0    0    0    0    0
</DCACConverter>

<ACACConverter>
@ idx name     i_node j_node r1   r2   control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat i_p i_q j_p j_q i_i j_i
# 0   acac_3_4 3      4      0.01 0.01 PQQ          5     0       0       0       0       1        0   0   0   0   0   0
</ACACConverter>
