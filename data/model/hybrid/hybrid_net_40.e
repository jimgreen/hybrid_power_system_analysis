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
#   1 load_3    4   1.0  50  30  20   1.0  40  40  20        1
#   2 load_4    5   1.0  60  20  20   1.0  50  30  20        1
#   3 load_5    6   1.0  40  40  20   1.0  30  40  30        1
#   4 load_6    7   1.0  50  30  20   1.0  50  30  20        1
#   5 load_7    8   1.0  70  20  10   1.0  60  20  20        1
#   6 load_8    9   1.0  50  30  20   1.0  40  30  30        1
#   7 load_9   10   1.0  60  20  20   1.0  50  30  20        1
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

<DCNode>
@ idx name  vbase voltage isl run_stat
#   1 nd_1  100.0     100   0        1
#   2 nd_2  100.0     100   0        1
#   3 nd_3  100.0     100   0        1
#   4 nd_4  100.0     100   0        1
#   5 nd_5  100.0     100   0        1
#   6 nd_6  100.0     100   0        1
#   7 nd_7  100.0     100   0        1
#   8 nd_8  100.0     100   0        1
#   9 nd_9  100.0     100   0        1
#  10 nd_10 100.0     100   0        1
#  11 nd_11 100.0     100   0        1
#  12 nd_12 100.0     100   0        1
#  13 nd_13 100.0     100   0        1
#  14 nd_14 100.0     100   0        1
#  15 nd_15 100.0     100   0        1
#  16 nd_16 100.0     100   0        1
#  17 nd_17 100.0     100   0        1
#  18 nd_18 100.0     100   0        1
#  19 nd_19 100.0     100   0        1
#  20 nd_20 100.0     100   0        1
#  21 nd_21 100.0     100   0        1
#  22 nd_22 100.0     100   0        1
#  23 nd_23 100.0     100   0        1
#  24 nd_24 100.0     100   0        1
#  25 nd_25 100.0     100   0        1
#  26 nd_26 100.0     100   0        1
#  27 nd_27 100.0     100   0        1
#  28 nd_28 100.0     100   0        1
#  29 nd_29 100.0     100   0        1
#  30 nd_30 100.0     100   0        1
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
#   1 line_0_1        1      2  0.1        1
#   2 line_1_2        2      3 0.02        1
#   3 line_2_3        3      4 0.03        1
#   4 line_3_4        4      5  0.1        1
#   5 line_4_0        5      1 0.02        1
#   6 line_5_6        6      7  0.1        1
#   7 line_6_7        7      8 0.02        1
#   8 line_7_8        8      9 0.03        1
#   9 line_8_9        9     10  0.1        1
#  10 line_9_5       10      6 0.02        1
#  11 line_10_11     11     12  0.1        1
#  12 line_11_12     12     13 0.02        1
#  13 line_12_13     13     14 0.03        1
#  14 line_13_14     14     15  0.1        1
#  15 line_14_10     15     11 0.02        1
#  16 line_15_16     16     17  0.1        1
#  17 line_16_17     17     18 0.02        1
#  18 line_17_18     18     19 0.03        1
#  19 line_18_19     19     20  0.1        1
#  20 line_19_15     20     16 0.02        1
#  21 line_4_5        5      6 0.04        1
#  22 line_9_10      10     11 0.05        1
#  23 line_14_15     15     16 0.03        1
#  24 line_2_6        3      7 0.06        1
#  25 line_8_12       9     13 0.07        1
#  26 line_13_17     14     18 0.05        1
#  27 line_0_19       1     20 0.08        1
#  28 line_20_21     21     22 0.02        1
#  29 line_21_22     22     23 0.03        1
#  30 line_22_23     23     24  0.1        1
#  31 line_23_24     24     25  0.2        1
#  32 line_24_20     25     21 0.04        1
#  33 line_25_26     26     27  0.1        1
#  34 line_26_27     27     28  0.2        1
#  35 line_27_28     28     29 0.03        1
#  36 line_28_29     29     30  0.1        1
#  37 line_29_25     30     26  0.2        1
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat
#   1 load_1     1   1.0 100  10  10        1
#   2 load_2     3   1.0 100  20   0        1
#   3 load_3     5   1.0 100  10  20        1
#   4 load_4     7   1.0 100  20   0        1
#   5 load_5     9   1.0 100  10  10        1
#   6 load_6    11   1.0 100  20   0        1
#   7 load_7    13   1.0 100  10  20        1
#   8 load_8    15   1.0 100  20   0        1
#   9 load_9    17   1.0 100  10  10        1
#  10 load_10   19   1.0 100  20   0        1
#  11 load_11   20   1.0 100  20  20        1
#  12 load_12   21   1.0 100  10   0        1
#  13 load_13   23   1.0 100  20  10        1
#  14 load_14   25   1.0 100  10   0        1
#  15 load_15   27   1.0 100  20  20        1
#  16 load_16   29   1.0 100  10   0        1
</DCLoad>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
#   1 gen_v1    4            V   160   100      0        1
#   2 gen_v2   11            P   100   100      0        1
#   3 gen_v3   18            P   100   150      0        1
#   4 gen_p1    6            P   110   200      0        1
#   5 gen_i1    8            I   110     0 0.0015        1
#   6 gen_p2   12            P   110   180      0        1
#   7 gen_i2   14            I   110     0 0.0012        1
#   8 gen_p3   16            P   110   220      0        1
#   9 gen_i3   20            I   110     0  0.001        1
#  10 gen_p4   22            P     0   150      0        1
#  11 gen_i4   24            I     0     0  0.001        1
#  12 gen_p5   26            V   140   200      0        1
#  13 gen_i5   28            I     0     0 0.0012        1
#  14 gen_p6   30            P     0   120      0        1
</DCGenerator>

<DCZeroBranch>
@ idx name i_node j_node run_stat
#   1 zbr_1_2        2      3        1
#   2 zbr_3_4        4      5        1
#   3 zbr_6_7        7      8        1
#   4 zbr_9_10      10     11        1
#   5 zbr_11_12     12     13        1
#   6 zbr_14_15     15     16        1
#   7 zbr_16_17     17     18        1
#   8 zbr_20_21     21     22        1
</DCZeroBranch>

<DCSwitch>
@ idx name i_node j_node status run_stat
#   1 sw_2_3        3      4      1        1
#   2 sw_6_8        7      9      1        1
#   3 sw_10_12     11     13      1        1
#   4 sw_15_17     16     18      1        1
</DCSwitch>

<DCBreak>
@ idx name i_node j_node status run_stat
#   1 sw_0_1        1      2      1        1
#   2 sw_4_5        5      6      1        1
#   3 sw_9_11      10     12      1        1
#   4 sw_13_14     14     15      1        1
#   5 sw_18_19     19     20      1        1
</DCBreak>

<DCDCConverter>
@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat
# 1 conv_1 5 6 0.05 0.05 CTRL_P SLACK 200 0 0 1
# 2 conv_2 9 10 0.1 0.1 CTRL_P SLACK 100 0 0 1
# 3 conv_3 13 14 0.075 0.075 CTRL_I SLACK 0 0.0015 0 1
# 4 conv_4 17 18 0.06 0.06 CTRL_P SLACK 150 0 0 1
# 5 conv_5 19 20 0.09 0.09 CTRL_I SLACK 0 0.001 0 1
# 6 conv_6 1 16 0.125 0.125 CTRL_P SLACK 100 0 0 1
# 7 conv_link 21 11 0.075 0.075 CTRL_V SLACK 0 0 120 1
# 8 conv_7 23 24 0.05 0.05 CTRL_P SLACK 120 0 0 1
# 9 conv_8 27 28 0.06 0.06 CTRL_I SLACK 0 0.0008 0 1
</DCDCConverter>

<DCACConverter>
@ idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat
# 1 inv_dcv 8 23 0.02 0.02 PQ V 0 0 0 117.2556 1
# 2 inv_acv 9 25 0.02 0.02 PH NONE 0 0 0.823466 0 1
# 3 inv_acp 10 29 0.02 0.02 PQ NONE -10 0 0 0 1
</DCACConverter>

<ACACConverter>
@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat
# 1 acac_3_4 4 5 0.01 0.01 Q Q 5 0 0 0 0 1
</ACACConverter>
