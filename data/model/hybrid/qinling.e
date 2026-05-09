<PowerBase>
@ p_base u_scale p_scale i_scale
# 100    1000.0  1.0     1000.0
</PowerBase>

<ACNode>
@ idx name        vbase voltage angle isl run_stat
# 0   wt01_src    300   300     0     0   1
# 1   wt02_src    300   300     0     0   1
# 2   wt03_src    300   300     0     0   1
# 3   wt04_src    300   300     0     0   1
# 4   wt05_src    300   300     0     0   1
# 5   wt06_src    300   300     0     0   1
# 6   wt07_src    300   300     0     0   1
# 7   wt08_src    300   300     0     0   1
# 8   wt09_src    300   300     0     0   1
# 9   wt10_src    300   300     0     0   1
# 10  wt01_rect   300   300     0     0   1
# 11  wt02_rect   300   300     0     0   1
# 12  wt03_rect   300   300     0     0   1
# 13  wt04_rect   300   300     0     0   1
# 14  wt05_rect   300   300     0     0   1
# 15  wt06_rect   300   300     0     0   1
# 16  wt07_rect   300   300     0     0   1
# 17  wt08_rect   300   300     0     0   1
# 18  wt09_rect   300   300     0     0   1
# 19  wt10_rect   300   300     0     0   1
# 20  ac_bus      380   380     0     0   1
# 21  diesel_node 380   380     0     0   1
# 22  ac_load_1   380   380     0     0   1
# 23  ac_load_2   380   380     0     0   1
# 24  grid_inv_ac 380   380     0     0   1
# 25  diesel_sw   380   380     0     0   1
# 26  load1_sw    380   380     0     0   1
# 27  load2_sw    380   380     0     0   1
# 28  grid_inv_sw 380   380     0     0   1
# 29  h2_load     380   380     0     0   1
# 30  h2_load_sw  380   380     0     0   1
</ACNode>

<ACBranch>
@ idx name         i_node j_node r     x     b   run_stat i_p i_q i_c j_p j_q j_c
# 0   wt01_cable   0      10     0.005 0.030 0.0 1        0   0   0   0   0   0
# 1   wt02_cable   1      11     0.005 0.030 0.0 1        0   0   0   0   0   0
# 2   wt03_cable   2      12     0.005 0.030 0.0 1        0   0   0   0   0   0
# 3   wt04_cable   3      13     0.005 0.030 0.0 1        0   0   0   0   0   0
# 4   wt05_cable   4      14     0.005 0.030 0.0 1        0   0   0   0   0   0
# 5   wt06_cable   5      15     0.005 0.030 0.0 1        0   0   0   0   0   0
# 6   wt07_cable   6      16     0.005 0.030 0.0 1        0   0   0   0   0   0
# 7   wt08_cable   7      17     0.005 0.030 0.0 1        0   0   0   0   0   0
# 8   wt09_cable   8      18     0.005 0.030 0.0 1        0   0   0   0   0   0
# 9   wt10_cable   9      19     0.005 0.030 0.0 1        0   0   0   0   0   0
# 10  diesel_line  21     25     0.001 0.005 0.0 1        0   0   0   0   0   0
# 11  load1_line   22     26     0.001 0.005 0.0 1        0   0   0   0   0   0
# 12  load2_line   23     27     0.001 0.005 0.0 1        0   0   0   0   0   0
# 13  inv_ac_line  24     28     0.001 0.005 0.0 1        0   0   0   0   0   0
# 14  h2_load_line 29     30     0.001 0.005 0.0 1        0   0   0   0   0   0
</ACBranch>

<ACLoad>
@ idx name      node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat p q current
# 0   load_ac_1 22   1.0   350 0   0   1.0   120 0   0   1        0 0 0
# 1   load_ac_2 23   1.0   250 0   0   1.0   80  0   0   1        0 0 0
# 2   h2_load   29   1.0   100 0   0   1.0   0   0   0   1        0 0 0
</ACLoad>

<ACGenerator>
@ idx name         node control_type p_set q_set v_set alpha run_stat p q current
# 0   wt01_10kw    0    V            0     0     300   1.0   1        0 0 0
# 1   wt02_10kw    1    V            0     0     300   1.0   1        0 0 0
# 2   wt03_10kw    2    V            0     0     300   1.0   1        0 0 0
# 3   wt04_10kw    3    V            0     0     300   1.0   1        0 0 0
# 4   wt05_10kw    4    V            0     0     300   1.0   1        0 0 0
# 5   wt06_10kw    5    V            0     0     300   1.0   1        0 0 0
# 6   wt07_10kw    6    V            0     0     300   1.0   1        0 0 0
# 7   wt08_10kw    7    V            0     0     300   1.0   1        0 0 0
# 8   wt09_10kw    8    V            0     0     300   1.0   1        0 0 0
# 9   wt10_10kw    9    V            0     0     300   1.0   1        0 0 0
# 10  diesel_300kw 21   V            0     0     380   1.0   1        0 0 0
</ACGenerator>

<ACSwitch>
@ idx name          i_node j_node status run_stat p q current
# 1   sw_load1_ac   20     26     1      1        0 0 0
# 3   sw_inv_ac     20     28     1      1        0 0 0
</ACSwitch>

<ACBreak>
@ idx name          i_node j_node status run_stat p q current
# 0   sw_diesel_ac  20     25     1      1        0 0 0
# 2   sw_load2_ac   20     27     1      1        0 0 0
# 4   sw_h2_load_ac 20     30     1      1        0 0 0
</ACBreak>

<DCNode>
@ idx name          vbase voltage isl run_stat
# 0   dc_bus_720v   720   720     0   1
# 1   wt01_dc_sw    720   720     0   1
# 2   wt02_dc_sw    720   720     0   1
# 3   wt03_dc_sw    720   720     0   1
# 4   wt04_dc_sw    720   720     0   1
# 5   wt05_dc_sw    720   720     0   1
# 6   wt06_dc_sw    720   720     0   1
# 7   wt07_dc_sw    720   720     0   1
# 8   wt08_dc_sw    720   720     0   1
# 9   wt09_dc_sw    720   720     0   1
# 10  wt10_dc_sw    720   720     0   1
# 11  pv01_300v     300   300     0   1
# 12  pv02_300v     300   300     0   1
# 13  pv03_300v     300   300     0   1
# 14  pv01_dc_sw    720   720     0   1
# 15  pv02_dc_sw    720   720     0   1
# 16  pv03_dc_sw    720   720     0   1
# 17  ess01_300v    300   300     0   1
# 18  ess02_300v    300   300     0   1
# 19  ess03_300v    300   300     0   1
# 20  ess04_300v    300   300     0   1
# 21  ess05_300v    300   300     0   1
# 22  ess01_720v    720   720     0   1
# 23  ess02_720v    720   720     0   1
# 24  ess03_720v    720   720     0   1
# 25  ess04_720v    720   720     0   1
# 26  ess05_720v    720   720     0   1
# 27  grid_inv_dc   720   720     0   1
# 28  wt01_line_dc  720   720     0   1
# 29  wt02_line_dc  720   720     0   1
# 30  wt03_line_dc  720   720     0   1
# 31  wt04_line_dc  720   720     0   1
# 32  wt05_line_dc  720   720     0   1
# 33  wt06_line_dc  720   720     0   1
# 34  wt07_line_dc  720   720     0   1
# 35  wt08_line_dc  720   720     0   1
# 36  wt09_line_dc  720   720     0   1
# 37  wt10_line_dc  720   720     0   1
# 38  pv01_line_dc  720   720     0   1
# 39  pv02_line_dc  720   720     0   1
# 40  pv03_line_dc  720   720     0   1
# 41  ess01_line_dc 720   720     0   1
# 42  ess02_line_dc 720   720     0   1
# 43  ess03_line_dc 720   720     0   1
# 44  ess04_line_dc 720   720     0   1
# 45  ess05_line_dc 720   720     0   1
# 46  inv_line_dc   720   720     0   1
# 47  fc01_src      720   720     0   1
# 48  fc01_line_dc  720   720     0   1
</DCNode>

<DCBranch>
@ idx name          i_node j_node r     run_stat i_p j_p current
# 0   wt01_dc_line  1      28     0.001 1        0   0   0
# 1   wt02_dc_line  2      29     0.001 1        0   0   0
# 2   wt03_dc_line  3      30     0.001 1        0   0   0
# 3   wt04_dc_line  4      31     0.001 1        0   0   0
# 4   wt05_dc_line  5      32     0.001 1        0   0   0
# 5   wt06_dc_line  6      33     0.001 1        0   0   0
# 6   wt07_dc_line  7      34     0.001 1        0   0   0
# 7   wt08_dc_line  8      35     0.001 1        0   0   0
# 8   wt09_dc_line  9      36     0.001 1        0   0   0
# 9   wt10_dc_line  10     37     0.001 1        0   0   0
# 10  pv01_dc_line  14     38     0.001 1        0   0   0
# 11  pv02_dc_line  15     39     0.001 1        0   0   0
# 12  pv03_dc_line  16     40     0.001 1        0   0   0
# 13  ess01_dc_line 22     41     0.001 1        0   0   0
# 14  ess02_dc_line 23     42     0.001 1        0   0   0
# 15  ess03_dc_line 24     43     0.001 1        0   0   0
# 16  ess04_dc_line 25     44     0.001 1        0   0   0
# 17  ess05_dc_line 26     45     0.001 1        0   0   0
# 18  inv_dc_line   27     46     0.001 1        0   0   0
# 19  fc01_dc_line  47     48     0.001 1        0   0   0
</DCBranch>

<DCGenerator>
@ idx name         node control_type v_set p_set i_set run_stat p current
# 0   dc_bus_vctrl 0    V            720   0     0     1        0 0
# 1   pv01_vsrc    11   V            300   0     0     1        0 0
# 2   pv02_vsrc    12   V            300   0     0     1        0 0
# 3   pv03_vsrc    13   V            300   0     0     1        0 0
# 4   ess01_vsrc   17   V            300   0     0     1        0 0
# 5   ess02_vsrc   18   V            300   0     0     1        0 0
# 6   ess03_vsrc   19   V            300   0     0     1        0 0
# 7   ess04_vsrc   20   V            300   0     0     1        0 0
# 8   ess05_vsrc   21   V            300   0     0     1        0 0
# 9   fc01_30kw    47   P            0     30    0     1        0 0
</DCGenerator>

<DCSwitch>
@ idx name        i_node j_node status run_stat p current
# 1   sw_wt02_dc  29     0      1      1        0 0
# 3   sw_wt04_dc  31     0      1      1        0 0
# 5   sw_wt06_dc  33     0      1      1        0 0
# 7   sw_wt08_dc  35     0      1      1        0 0
# 9   sw_wt10_dc  37     0      1      1        0 0
# 11  sw_pv02_dc  39     0      1      1        0 0
# 13  sw_ess01_dc 41     0      1      1        0 0
# 15  sw_ess03_dc 43     0      1      1        0 0
# 17  sw_ess05_dc 45     0      1      1        0 0
# 19  sw_fc01_dc  48     0      1      1        0 0
</DCSwitch>

<DCBreak>
@ idx name        i_node j_node status run_stat p current
# 0   sw_wt01_dc  28     0      1      1        0 0
# 2   sw_wt03_dc  30     0      1      1        0 0
# 4   sw_wt05_dc  32     0      1      1        0 0
# 6   sw_wt07_dc  34     0      1      1        0 0
# 8   sw_wt09_dc  36     0      1      1        0 0
# 10  sw_pv01_dc  38     0      1      1        0 0
# 12  sw_pv03_dc  40     0      1      1        0 0
# 14  sw_ess02_dc 42     0      1      1        0 0
# 16  sw_ess04_dc 44     0      1      1        0 0
# 18  sw_grid_dc  46     0      1      1        0 0
</DCBreak>

<DCDCConverter>
@ idx name       i_node j_node r1    r2    control_type p_set i_set v_set run_stat i_p j_p i_c j_c
# 0   pv01_dcdc  11     14     0.005 0.005 P            50    0     0     1        0   0   0   0
# 1   pv02_dcdc  12     15     0.005 0.005 P            50    0     0     1        0   0   0   0
# 2   pv03_dcdc  13     16     0.005 0.005 P            30    0     0     1        0   0   0   0
# 3   ess01_dcdc 17     22     0.005 0.005 P            60    0     0     1        0   0   0   0
# 4   ess02_dcdc 18     23     0.005 0.005 P            60    0     0     1        0   0   0   0
# 5   ess03_dcdc 19     24     0.005 0.005 P            60    0     0     1        0   0   0   0
# 6   ess04_dcdc 20     25     0.005 0.005 P            60    0     0     1        0   0   0   0
# 7   ess05_dcdc 21     26     0.005 0.005 P            60    0     0     1        0   0   0   0
</DCDCConverter>

<DCACConverter>
@ idx name         ac_node dc_node r1    r2    control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat dc_p ac_p ac_q dc_i ac_i
# 0   wt01_rect    10      1       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 1   wt02_rect    11      2       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 2   wt03_rect    12      3       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 3   wt04_rect    13      4       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 4   wt05_rect    14      5       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 5   wt06_rect    15      6       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 6   wt07_rect    16      7       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 7   wt08_rect    17      8       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 8   wt09_rect    18      9       0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 9   wt10_rect    19      10      0.005 0.005 ACP          10       0        0        0        1        0    0    0    0    0
# 10  grid_inv_acp 24      27      0.005 0.005 ACP          -350     0        0        0        1        0    0    0    0    0
</DCACConverter>
