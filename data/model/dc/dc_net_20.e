<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV kW kA
</PowerBase>

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
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
#   1 line_0_1        1      2 0.01        1
#   2 line_1_2        2      3 0.02        1
#   3 line_2_3        3      4 0.03        1
#   4 line_3_4        4      5 0.01        1
#   5 line_4_0        5      1 0.02        1
#   6 line_5_6        6      7 0.01        1
#   7 line_6_7        7      8 0.02        1
#   8 line_7_8        8      9 0.03        1
#   9 line_8_9        9     10 0.01        1
#  10 line_9_5       10      6 0.02        1
#  11 line_10_11     11     12 0.01        1
#  12 line_11_12     12     13 0.02        1
#  13 line_12_13     13     14 0.03        1
#  14 line_13_14     14     15 0.01        1
#  15 line_14_10     15     11 0.02        1
#  16 line_15_16     16     17 0.01        1
#  17 line_16_17     17     18 0.02        1
#  18 line_17_18     18     19 0.03        1
#  19 line_18_19     19     20 0.01        1
#  20 line_19_15     20     16 0.02        1
#  21 line_4_5        5      6 0.04        1
#  22 line_9_10      10     11 0.05        1
#  23 line_14_15     15     16 0.03        1
#  24 line_2_6        3      7 0.06        1
#  25 line_8_12       9     13 0.07        1
#  26 line_13_17     14     18 0.05        1
#  27 line_0_19       1     20 0.08        1
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat
#   1 load_1     1   1.0 200   0   0        1
#   2 load_2     3   1.0   0   0 100        1
#   3 load_3     5   1.0   0 150   0        1
#   4 load_4     7   1.0 150   0   0        1
#   5 load_5     9   1.0  10  10  90        1
#   6 load_6    11   1.0 100   0   0        1
#   7 load_7    13   1.0   0 120   0        1
#   8 load_8    15   1.0  20  20  80        1
#   9 load_9    17   1.0 250   0   0        1
#  10 load_10   19   1.0   0  90   0        1
#  11 load_11   20   1.0 120   0   0        1
</DCLoad>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
#   1 gen_v1    4            V   150     0      0        1
#   2 gen_v2   11            P     0   100      0        1
#   3 gen_v3   18            P     0   150      0        1
#   4 gen_p1    6            P   110   200      0        1
#   5 gen_i1    8            I   110     0 0.0015        1
#   6 gen_p2   12            P   110   180      0        1
#   7 gen_i2   14            I   110     0 0.0012        1
#   8 gen_p3   16            P   110   220      0        1
#   9 gen_i3   20            I   110     0  0.001        1
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
@ idx name i_node j_node r1 r2 control_type p_set i_set v_set run_stat
#   1 conv_1      5      6  0.05  0.05            P   200      0     0        1
#   2 conv_2      9     10   0.1   0.1            P   100      0     0        1
#   3 conv_3     13     14 0.075 0.075            I     0 0.0015     0        1
#   4 conv_4     17     18  0.06  0.06            P   150      0     0        1
#   5 conv_5     19     20  0.09  0.09            I     0  0.001     0        1
#   6 conv_6      1     16 0.125 0.125            P   100      0     0        1
</DCDCConverter>
