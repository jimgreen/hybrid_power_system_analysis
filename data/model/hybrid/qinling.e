<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1000.0  1.0     1000.0
</PowerBase>

<ACNode>
@ idx name        vbase voltage angle isl run_stat
#   1 wt01_src      300     300     0   0        1
#   2 wt02_src      300     300     0   0        1
#   3 wt03_src      300     300     0   0        1
#   4 wt04_src      300     300     0   0        1
#   5 wt05_src      300     300     0   0        1
#   6 wt06_src      300     300     0   0        1
#   7 wt07_src      300     300     0   0        1
#   8 wt08_src      300     300     0   0        1
#   9 wt09_src      300     300     0   0        1
#  10 wt10_src      300     300     0   0        1
#  11 wt01_rect     300     300     0   0        1
#  12 wt02_rect     300     300     0   0        1
#  13 wt03_rect     300     300     0   0        1
#  14 wt04_rect     300     300     0   0        1
#  15 wt05_rect     300     300     0   0        1
#  16 wt06_rect     300     300     0   0        1
#  17 wt07_rect     300     300     0   0        1
#  18 wt08_rect     300     300     0   0        1
#  19 wt09_rect     300     300     0   0        1
#  20 wt10_rect     300     300     0   0        1
#  21 ac_bus        380     380     0   0        1
#  22 diesel_node   380     380     0   0        1
#  23 ac_load_1     380     380     0   0        1
#  24 ac_load_2     380     380     0   0        1
#  25 grid_inv_ac   380     380     0   0        1
#  26 diesel_sw     380     380     0   0        1
#  27 load1_sw      380     380     0   0        1
#  28 load2_sw      380     380     0   0        1
#  29 grid_inv_sw   380     380     0   0        1
#  30 h2_load       380     380     0   0        1
#  31 h2_load_sw    380     380     0   0        1
</ACNode>

<ACBranch>
@ idx name i_node j_node r x b run_stat
#   1 wt01_cable        1     11 0.005 0.030 0.0        1
#   2 wt02_cable        2     12 0.005 0.030 0.0        1
#   3 wt03_cable        3     13 0.005 0.030 0.0        1
#   4 wt04_cable        4     14 0.005 0.030 0.0        1
#   5 wt05_cable        5     15 0.005 0.030 0.0        1
#   6 wt06_cable        6     16 0.005 0.030 0.0        1
#   7 wt07_cable        7     17 0.005 0.030 0.0        1
#   8 wt08_cable        8     18 0.005 0.030 0.0        1
#   9 wt09_cable        9     19 0.005 0.030 0.0        1
#  10 wt10_cable       10     20 0.005 0.030 0.0        1
#  11 diesel_line      22     26 0.001 0.005 0.0        1
#  12 load1_line       23     27 0.001 0.005 0.0        1
#  13 load2_line       24     28 0.001 0.005 0.0        1
#  14 inv_ac_line      25     29 0.001 0.005 0.0        1
#  15 h2_load_line     30     31 0.001 0.005 0.0        1
</ACBranch>

<ACLoad>
@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat
#   1 load_ac_1   23   1.0 350   0   0   1.0 120   0   0        1
#   2 load_ac_2   24   1.0 250   0   0   1.0  80   0   0        1
#   3 h2_load     30   1.0 100   0   0   1.0   0   0   0        1
</ACLoad>

<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha run_stat
#   1 wt01_10kw       1            V     0     0   300   1.0        1
#   2 wt02_10kw       2            V     0     0   300   1.0        1
#   3 wt03_10kw       3            V     0     0   300   1.0        1
#   4 wt04_10kw       4            V     0     0   300   1.0        1
#   5 wt05_10kw       5            V     0     0   300   1.0        1
#   6 wt06_10kw       6            V     0     0   300   1.0        1
#   7 wt07_10kw       7            V     0     0   300   1.0        1
#   8 wt08_10kw       8            V     0     0   300   1.0        1
#   9 wt09_10kw       9            V     0     0   300   1.0        1
#  10 wt10_10kw      10            V     0     0   300   1.0        1
#  11 diesel_300kw   22            V     0     0   380   1.0        1
</ACGenerator>

<ACSwitch>
@ idx name i_node j_node status run_stat
#   1 sw_load1_ac     21     27      1        1
#   2 sw_inv_ac       21     29      1        1
</ACSwitch>

<ACBreak>
@ idx name i_node j_node status run_stat
#   1 sw_diesel_ac      21     26      1        1
#   2 sw_load2_ac       21     28      1        1
#   3 sw_h2_load_ac     21     31      1        1
</ACBreak>

<DCNode>
@ idx name          vbase voltage isl run_stat
#   1 dc_bus_720v     720     720   0        1
#   2 wt01_dc_sw      720     720   0        1
#   3 wt02_dc_sw      720     720   0        1
#   4 wt03_dc_sw      720     720   0        1
#   5 wt04_dc_sw      720     720   0        1
#   6 wt05_dc_sw      720     720   0        1
#   7 wt06_dc_sw      720     720   0        1
#   8 wt07_dc_sw      720     720   0        1
#   9 wt08_dc_sw      720     720   0        1
#  10 wt09_dc_sw      720     720   0        1
#  11 wt10_dc_sw      720     720   0        1
#  12 pv01_300v       300     300   0        1
#  13 pv02_300v       300     300   0        1
#  14 pv03_300v       300     300   0        1
#  15 pv01_dc_sw      720     720   0        1
#  16 pv02_dc_sw      720     720   0        1
#  17 pv03_dc_sw      720     720   0        1
#  18 ess01_300v      300     300   0        1
#  19 ess02_300v      300     300   0        1
#  20 ess03_300v      300     300   0        1
#  21 ess04_300v      300     300   0        1
#  22 ess05_300v      300     300   0        1
#  23 ess01_720v      720     720   0        1
#  24 ess02_720v      720     720   0        1
#  25 ess03_720v      720     720   0        1
#  26 ess04_720v      720     720   0        1
#  27 ess05_720v      720     720   0        1
#  28 grid_inv_dc     720     720   0        1
#  29 wt01_line_dc    720     720   0        1
#  30 wt02_line_dc    720     720   0        1
#  31 wt03_line_dc    720     720   0        1
#  32 wt04_line_dc    720     720   0        1
#  33 wt05_line_dc    720     720   0        1
#  34 wt06_line_dc    720     720   0        1
#  35 wt07_line_dc    720     720   0        1
#  36 wt08_line_dc    720     720   0        1
#  37 wt09_line_dc    720     720   0        1
#  38 wt10_line_dc    720     720   0        1
#  39 pv01_line_dc    720     720   0        1
#  40 pv02_line_dc    720     720   0        1
#  41 pv03_line_dc    720     720   0        1
#  42 ess01_line_dc   720     720   0        1
#  43 ess02_line_dc   720     720   0        1
#  44 ess03_line_dc   720     720   0        1
#  45 ess04_line_dc   720     720   0        1
#  46 ess05_line_dc   720     720   0        1
#  47 inv_line_dc     720     720   0        1
#  48 fc01_src        720     720   0        1
#  49 fc01_line_dc    720     720   0        1
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
#   1 wt01_dc_line       2     29 0.001        1
#   2 wt02_dc_line       3     30 0.001        1
#   3 wt03_dc_line       4     31 0.001        1
#   4 wt04_dc_line       5     32 0.001        1
#   5 wt05_dc_line       6     33 0.001        1
#   6 wt06_dc_line       7     34 0.001        1
#   7 wt07_dc_line       8     35 0.001        1
#   8 wt08_dc_line       9     36 0.001        1
#   9 wt09_dc_line      10     37 0.001        1
#  10 wt10_dc_line      11     38 0.001        1
#  11 pv01_dc_line      15     39 0.001        1
#  12 pv02_dc_line      16     40 0.001        1
#  13 pv03_dc_line      17     41 0.001        1
#  14 ess01_dc_line     23     42 0.001        1
#  15 ess02_dc_line     24     43 0.001        1
#  16 ess03_dc_line     25     44 0.001        1
#  17 ess04_dc_line     26     45 0.001        1
#  18 ess05_dc_line     27     46 0.001        1
#  19 inv_dc_line       28     47 0.001        1
#  20 fc01_dc_line      48     49 0.001        1
</DCBranch>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
#   1 dc_bus_vctrl    1            V   720     0     0        1
#   2 pv01_vsrc      12            V   300     0     0        1
#   3 pv02_vsrc      13            V   300     0     0        1
#   4 pv03_vsrc      14            V   300     0     0        1
#   5 ess01_vsrc     18            V   300     0     0        1
#   6 ess02_vsrc     19            V   300     0     0        1
#   7 ess03_vsrc     20            V   300     0     0        1
#   8 ess04_vsrc     21            V   300     0     0        1
#   9 ess05_vsrc     22            V   300     0     0        1
#  10 fc01_30kw      48            P     0    30     0        1
</DCGenerator>

<DCSwitch>
@ idx name i_node j_node status run_stat
#   1 sw_wt02_dc      30      1      1        1
#   2 sw_wt04_dc      32      1      1        1
#   3 sw_wt06_dc      34      1      1        1
#   4 sw_wt08_dc      36      1      1        1
#   5 sw_wt10_dc      38      1      1        1
#   6 sw_pv02_dc      40      1      1        1
#   7 sw_ess01_dc     42      1      1        1
#   8 sw_ess03_dc     44      1      1        1
#   9 sw_ess05_dc     46      1      1        1
#  10 sw_fc01_dc      49      1      1        1
</DCSwitch>

<DCBreak>
@ idx name i_node j_node status run_stat
#   1 sw_wt01_dc      29      1      1        1
#   2 sw_wt03_dc      31      1      1        1
#   3 sw_wt05_dc      33      1      1        1
#   4 sw_wt07_dc      35      1      1        1
#   5 sw_wt09_dc      37      1      1        1
#   6 sw_pv01_dc      39      1      1        1
#   7 sw_pv03_dc      41      1      1        1
#   8 sw_ess02_dc     43      1      1        1
#   9 sw_ess04_dc     45      1      1        1
#  10 sw_grid_dc      47      1      1        1
</DCBreak>

<DCDCConverter>
@ idx name i_node j_node r1 r2 control_type p_set i_set v_set run_stat
#   1 pv01_dcdc      12     15 0.005 0.005            P    50     0     0        1
#   2 pv02_dcdc      13     16 0.005 0.005            P    50     0     0        1
#   3 pv03_dcdc      14     17 0.005 0.005            P    30     0     0        1
#   4 ess01_dcdc     18     23 0.005 0.005            P    60     0     0        1
#   5 ess02_dcdc     19     24 0.005 0.005            P    60     0     0        1
#   6 ess03_dcdc     20     25 0.005 0.005            P    60     0     0        1
#   7 ess04_dcdc     21     26 0.005 0.005            P    60     0     0        1
#   8 ess05_dcdc     22     27 0.005 0.005            P    60     0     0        1
</DCDCConverter>

<DCACConverter>
@ idx name ac_node dc_node r1 r2 control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat
#   1 wt01_rect         11       2 0.005 0.005          ACP       10        0        0        0        1
#   2 wt02_rect         12       3 0.005 0.005          ACP       10        0        0        0        1
#   3 wt03_rect         13       4 0.005 0.005          ACP       10        0        0        0        1
#   4 wt04_rect         14       5 0.005 0.005          ACP       10        0        0        0        1
#   5 wt05_rect         15       6 0.005 0.005          ACP       10        0        0        0        1
#   6 wt06_rect         16       7 0.005 0.005          ACP       10        0        0        0        1
#   7 wt07_rect         17       8 0.005 0.005          ACP       10        0        0        0        1
#   8 wt08_rect         18       9 0.005 0.005          ACP       10        0        0        0        1
#   9 wt09_rect         19      10 0.005 0.005          ACP       10        0        0        0        1
#  10 wt10_rect         20      11 0.005 0.005          ACP       10        0        0        0        1
#  11 grid_inv_acp      25      28 0.005 0.005          ACP     -350        0        0        0        1
</DCACConverter>
