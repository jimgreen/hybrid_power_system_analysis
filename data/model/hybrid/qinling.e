<Model>
@    path           name       p_base    u_unit    p_unit    i_unit
#    IEEE标准算例    qinling    100       V         kW        A
</Model>

<basevoltage>
@    idx    name    vltp
#    1      0       0
#    2      0.4     0.4
#    3      6       6
#    4      10      10
#    5      10.5    10.5
#    6      35      35
#    7      66      66
#    8      110     110
#    9      220     220
#    10     330     330
#    11     500     500
#    12     750     750
#    13     800     800
#    14     0       0
#    15     0.4     0.4
#    16     6       6
#    17     10      10
#    18     10.5    10.5
#    19     35      35
#    20     66      66
#    21     110     110
#    22     220     220
#    23     330     330
#    24     500     500
#    25     750     750
#    26     800     800
</basevoltage>

<ACNode>
@    idx    name                  vbase    run_stat
#    1      交流风电-1             380      1
#    2      交流风电-2             380      1
#    3      交流风电-3             380      1
#    4      交流风电-4             380      1
#    5      交流风电-5             380      1
#    6      交流风电-6             380      1
#    7      交流风电-7             380      1
#    8      交流风电-8             380      1
#    9      交流风电-9             380      1
#    10     交流风电-10            380      1
#    11     风机变流器-1            380      1
#    12     风机变流器-2            380      1
#    13     风机变流器-3            380      1
#    14     风机变流器-4            380      1
#    15     风机变流器-5            380      1
#    16     风机变流器-6            380      1
#    17     风机变流器-7            380      1
#    18     风机变流器-8            380      1
#    19     风机变流器-9            380      1
#    20     风机变流器-10           380      1
#    21     交流线路（自适应）-30    380      1
#    22     交流线路（自适应）-30    380      1
#    23     DCAC变流器-1           380      1
#    24     交流母线（竖向）-1       380      1
#    25     DCAC变流器-2           380      1
#    26     交流柴油发电机-27       380      1
#    27     交流柴油发电机-28       380      1
#    28     交流柴油发电机-29       380      1
#    29     交流柴油发电机-30       380      1
#    30     盒型开关-7             380      1
#    31     盒型开关-9             380      1
#    32     交流负荷-1             380      1
#    33     交流电化学储能-23       380      1
#    34     交流风力发电机-24       380      1
#    35     交流光伏发电机-25       380      1
#    36     交流电化学储能-26       380      1
</ACNode>

<ACRealBs>
@    idx    name               dev_type           node    run_stat    v_max    v_min
#    1      交流母线（竖向）-1    ac-bus-vertical    24      1           1.1      0.9
</ACRealBs>

<ACBranch>
@    idx    name                  dev_type            i_node    j_node    run_stat    rated_capacity    i_max    r      x      b
#    20     交流线路（自适应）-20    ac-routable-line    1         11        1           0                 0        0.1    1.0    0.0
#    21     交流线路（自适应）-21    ac-routable-line    2         12        1           0                 0        0.1    1.0    0.0
#    22     交流线路（自适应）-22    ac-routable-line    3         13        1           0                 0        0.1    1.0    0.0
#    23     交流线路（自适应）-23    ac-routable-line    4         14        1           0                 0        0.1    1.0    0.0
#    24     交流线路（自适应）-24    ac-routable-line    5         15        1           0                 0        0.1    1.0    0.0
#    25     交流线路（自适应）-25    ac-routable-line    6         16        1           0                 0        0.1    1.0    0.0
#    26     交流线路（自适应）-26    ac-routable-line    7         17        1           0                 0        0.1    1.0    0.0
#    27     交流线路（自适应）-27    ac-routable-line    8         18        1           0                 0        0.1    1.0    0.0
#    28     交流线路（自适应）-28    ac-routable-line    9         19        1           0                 0        0.1    1.0    0.0
#    29     交流线路（自适应）-29    ac-routable-line    10        20        1           0                 0        0.1    1.0    0.0
#    30     交流线路（自适应）-30    ac-routable-line    21        22        1           0                 0        0.1    1.0    0.0
</ACBranch>

<ACLoad>
@    idx    name                            dev_type           node    p_max    p_min    q_max    q_min    run_stat    rated_capacity    pbase    pv0    pv1    pv2    qbase    qv0    qv1    qv2    v_max    v_min
#    1      交流负荷-1                       ac-load            32      5        0        1.2      0        1           5                 150      1.0    0.0    0.0    50       1.0    0.0    0.0    1.1      0.9
#    2      交流电制氢-1_交流设备端交流电负荷    ac-electrolyzer    31      0        0        0        0        1           0                 0        1.0    0.0    0.0    0        1.0    0.0    0.0    0        0
</ACLoad>

<ACGenerator>
@    idx    name               dev_type            node    control_type    p_set    p_max    p_min    q_set    q_max    q_min    v_set    alpha    run_stat    rated_capacity    rated_voltage    v_max    v_min
#    1      交流风电-1          ac-wind-source      1       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    2      交流风电-2          ac-wind-source      2       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    3      交流风电-3          ac-wind-source      3       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    4      交流风电-4          ac-wind-source      4       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    5      交流风电-5          ac-wind-source      5       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    6      交流风电-6          ac-wind-source      6       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    7      交流风电-7          ac-wind-source      7       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    8      交流风电-8          ac-wind-source      8       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    9      交流风电-9          ac-wind-source      9       PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    10     交流风电-10         ac-wind-source      10      PQ              3.0      0        0        0.5      0        0        380      0.5      1           10.1              380              1.1      0.9
#    23     交流电化学储能-23    ac-storage          33      PQ              0.0      100      -100     0.0      0        0        380      1.0      1           100               380              1.1      0.9
#    24     交流风力发电机-24    ac-wind-source      34      PQ              0        50       0        0        0        0        380      1.0      1           50                380              1.1      0.9
#    25     交流光伏发电机-25    ac-pv-source        35      PV              0        0        0        0        0        0        380      1.0      1           20                10               1.1      0.9
#    26     交流电化学储能-26    ac-storage          36      PH              0.0      100      -100     0.0      0        0        380      1.0      1           5                 380              1.1      0.9
#    27     交流柴油发电机-27    ac-diesel-source    26      PH              0        300      70       0        0        0        380      1.0      1           300               380              1.1      0.9
#    28     交流柴油发电机-28    ac-diesel-source    27      PH              0        300      70       0        0        0        380      1.0      1           300               380              1.1      0.9
#    29     交流柴油发电机-29    ac-diesel-source    28      PH              0        300      70       0        0        0        380      1.0      1           300               380              1.1      0.9
#    30     交流柴油发电机-30    ac-diesel-source    29      PH              0        300      70       0        0        0        380      1.0      1           300               380              1.1      0.9
</ACGenerator>

<ACZeroBranch>
@    idx    name                      dev_type                   i_node    j_node    run_stat
#    1      交流零阻抗支路（自适应）-1    ac-zero-routable-branch    30        32        1
</ACZeroBranch>

<ACBreak>
@    idx    name          dev_type          i_node    j_node    status    run_stat    rated_capacity    i_max
#    1      交流断路器-1    ac-breaker        23        24        1         1           0                 0
#    2      交流断路器-2    ac-breaker        25        24        1         1           0                 0
#    3      盒型开关-3      ac-box-breaker    24        26        1         1           0                 0
#    4      盒型开关-4      ac-box-breaker    24        27        1         1           0                 0
#    5      盒型开关-5      ac-box-breaker    24        28        1         1           0                 0
#    6      盒型开关-6      ac-box-breaker    24        29        1         1           0                 0
#    7      盒型开关-7      ac-box-breaker    24        30        1         1           0                 0
#    8      盒型开关-8      ac-box-breaker    24        21        1         1           0                 0
#    9      盒型开关-9      ac-box-breaker    31        22        1         1           0                 0
#    10     盒型开关-10     ac-box-breaker    24        33        1         1           0                 0
#    11     盒型开关-11     ac-box-breaker    24        34        1         1           0                 0
#    12     盒型开关-12     ac-box-breaker    24        35        1         1           0                 0
#    13     盒型开关-13     ac-box-breaker    36        24        1         1           0                 0
</ACBreak>

<DCNode>
@    idx    name               vbase    voltage    isl    run_stat
#    1      风机变流器-1         750      750        0      1
#    2      风机变流器-2         750      750        0      1
#    3      风机变流器-3         750      750        0      1
#    4      风机变流器-4         750      750        0      1
#    5      风机变流器-5         750      750        0      1
#    6      风机变流器-6         750      750        0      1
#    7      风机变流器-7         750      750        0      1
#    8      风机变流器-8         750      750        0      1
#    9      风机变流器-9         750      750        0      1
#    10     风机变流器-10        750      750        0      1
#    11     直流母线（竖向）-1    750      750        0      1
#    12     光伏变流器-1         750      750        0      1
#    13     光伏变流器-2         750      750        0      1
#    14     光伏变流器-3         750      750        0      1
#    15     储能变流器-1         750      750        0      1
#    16     储能变流器-2         750      750        0      1
#    17     储能变流器-3         750      750        0      1
#    18     储能变流器-4         750      750        0      1
#    19     储能变流器-5         750      750        0      1
#    20     储能变流器-6         750      750        0      1
#    21     DCAC变流器-1        750      750        0      1
#    22     DCAC变流器-2        750      750        0      1
#    23     直流断路器-31        750      750        0      1
#    24     直流断路器-32        750      750        0      1
#    25     直流断路器-32        750      750        0      1
#    26     直流光伏-1           400      400        0      1
#    27     直流光伏-2           400      400        0      1
#    28     直流光伏-3           400      400        0      1
#    29     电化学储能-1         500      500        0      1
#    30     电化学储能-2         500      500        0      1
#    31     电化学储能-3         500      500        0      1
#    32     电化学储能-4         500      500        0      1
#    33     电化学储能-5         500      500        0      1
#    34     电化学储能-6         500      500        0      1
#    35     光伏变流器-1         400      400        0      1
#    36     光伏变流器-2         400      400        0      1
#    37     光伏变流器-3         400      400        0      1
#    38     直流负荷-1           750      750        0      1
</DCNode>

<DCRealBs>
@    idx    name               dev_type           node    run_stat    v_max    v_min
#    1      直流母线（竖向）-1    dc-bus-vertical    11      1           1.1      0.9
</DCRealBs>

<DCBranch>
@    idx    name               dev_type            i_node    j_node    run_stat    rated_capacity    i_max    r
#    1      光伏直流线路-1       dc-routable-line    26        35        1           0                 0        1.0
#    2      光伏直流线路-2       dc-routable-line    27        36        1           0                 0        1.0
#    3      光伏直流线路-3       dc-routable-line    28        37        1           0                 0        1.0
#    4      燃料电池直流线路-1    dc-routable-line    23        24        1           0                 0        1.0
</DCBranch>

<DCLoad>
@    idx    name         dev_type    node    p_max    p_min    run_stat    rated_capacity    pbase    pv0    pv1    pv2    v_max    v_min
#    1      直流负荷-1    dc-load     38      1.5      0        1           1.5               0        1.0    0.0    0.0    1.1      0.9
</DCLoad>

<DCGenerator>
@    idx    name                            dev_type        node    control_type    v_set    p_set    p_max    p_min    i_set    run_stat    rated_capacity    rated_voltage    v_max    v_min
#    1      直流光伏-1                       dc-pv-source    26      P               400      5.0      0        0        0.0      1           50                400              1.1      0.9
#    2      直流光伏-2                       dc-pv-source    27      P               400      5.0      0        0        0.0      1           50                400              1.1      0.9
#    3      直流光伏-3                       dc-pv-source    28      P               400      5.0      0        0        0.0      1           50                400              1.1      0.9
#    4      电化学储能-1                      dc-storage      29      P               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    5      电化学储能-2                      dc-storage      30      P               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    6      电化学储能-3                      dc-storage      31      V               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    7      电化学储能-4                      dc-storage      32      V               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    8      电化学储能-5                      dc-storage      33      V               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    9      电化学储能-6                      dc-storage      34      V               500      0.0      0        0        0.0      1           60                500              1.1      0.9
#    10     直流燃料电池-1_直流设备端直流电源    dc-fuel-cell    25      P               750      0        0        0        0        1           0                 0                0        0
</DCGenerator>

<DCBreak>
@    idx    name           dev_type      i_node    j_node    status    run_stat    rated_capacity    i_max
#    1      直流断路器-1     dc-breaker    1         11        1         1           1600              1600
#    2      直流断路器-2     dc-breaker    2         11        1         1           1600              1600
#    3      直流断路器-3     dc-breaker    3         11        1         1           1600              1600
#    4      直流断路器-4     dc-breaker    4         11        1         1           1600              1600
#    5      直流断路器-5     dc-breaker    5         11        1         1           1600              1600
#    6      直流断路器-6     dc-breaker    6         11        1         1           1600              1600
#    7      直流断路器-7     dc-breaker    7         11        1         1           1600              1600
#    8      直流断路器-8     dc-breaker    8         11        1         1           1600              1600
#    9      直流断路器-9     dc-breaker    9         11        1         1           1600              1600
#    11     直流断路器-11    dc-breaker    10        11        1         1           1600              1600
#    12     直流断路器-12    dc-breaker    12        11        1         1           1600              1600
#    13     直流断路器-13    dc-breaker    13        11        1         1           1600              1600
#    14     直流断路器-14    dc-breaker    14        11        1         1           1600              1600
#    15     直流断路器-15    dc-breaker    11        15        1         1           1600              1600
#    16     直流断路器-16    dc-breaker    11        16        1         1           1600              1600
#    17     直流断路器-17    dc-breaker    11        17        1         1           1600              1600
#    18     直流断路器-18    dc-breaker    11        18        1         1           1600              1600
#    20     直流断路器-20    dc-breaker    11        19        1         1           1600              1600
#    21     直流断路器-21    dc-breaker    11        20        1         1           1600              1600
#    29     直流断路器-29    dc-breaker    11        21        1         1           1600              1600
#    30     直流断路器-30    dc-breaker    11        22        1         1           1600              1600
#    31     直流断路器-31    dc-breaker    11        23        1         1           1600              1600
#    32     直流断路器-32    dc-breaker    24        25        1         1           1600              1600
#    33     直流断路器-33    dc-breaker    11        38        1         1           1600              1600
</DCBreak>

<DCDCConverter>
@    idx    name          dev_type          i_node    j_node    i_control_type    j_control_type    p_set    i_set    v_set    run_stat    rated_capacity    i_p_max    i_p_min    i_i_max    i_v_max    i_v_min    j_p_max    j_p_min    j_i_max    j_v_max    j_v_min    r1    r2
#    1      光伏变流器-1    dcdc-converter    35        12        V                 NONE              0        0        400      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    2      光伏变流器-2    dcdc-converter    36        13        V                 NONE              0        0        400      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    3      光伏变流器-3    dcdc-converter    37        14        V                 NONE              0        0        400      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    4      储能变流器-1    dcdc-converter    15        29        NONE              V                 0        0        400      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    5      储能变流器-2    dcdc-converter    16        30        NONE              V                 0        0        400      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    6      储能变流器-3    dcdc-converter    17        31        V                 NONE              0        0        750      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    7      储能变流器-4    dcdc-converter    18        32        V                 NONE              0        0        750      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    8      储能变流器-5    dcdc-converter    19        33        V                 NONE              0        0        750      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
#    9      储能变流器-6    dcdc-converter    20        34        V                 NONE              0        0        750      1           5                 5          -5         0          1.1        0.9        5          -5         0          1.1        0.9        0     0
</DCDCConverter>

<DCACConverter>
@    idx    name           dev_type          ac_node    dc_node    ac_control_type    dc_control_type    p_ac_set    q_ac_set    v_ac_set    p_dc_set    v_dc_set    run_stat    rated_capacity    ac_p_max    ac_p_min    ac_i_max    ac_v_max    ac_v_min    dc_p_max    dc_p_min    dc_i_max    dc_v_max    dc_v_min    r1    r2
#    1      风机变流器-1     acdc-converter    11         1          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    2      风机变流器-2     acdc-converter    12         2          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    3      风机变流器-3     acdc-converter    13         3          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    4      风机变流器-4     acdc-converter    14         4          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    5      风机变流器-5     acdc-converter    15         5          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    6      风机变流器-6     acdc-converter    16         6          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    7      风机变流器-7     acdc-converter    17         7          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    8      风机变流器-8     acdc-converter    18         8          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    9      风机变流器-9     acdc-converter    19         9          PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    10     风机变流器-10    acdc-converter    20         10         PH                 NONE               0           0           380         0           750         1           10                10          -10         0           1.1         0.9         10          -10         0           1.1         0.9         0     0
#    13     DCAC变流器-1    dcac-converter    23         21         PQ                 NONE               0           0           380         0           750         1           300               300         -300        0           1.1         0.9         300         -300        0           1.1         0.9         0     0
#    14     DCAC变流器-2    dcac-converter    25         22         PQ                 NONE               0           0           380         0           750         1           300               300         -300        0           1.1         0.9         300         -300        0           1.1         0.9         0     0
</DCACConverter>

<HydroMedium>
@    density    compressibility    molar_mass    temperature    flow_factor
#    0.08375    1.0                0.002016      288.15         0.35
</HydroMedium>

<HydroNode>
@    idx    name                    pressure    run_stat
#    1      电制氢出口节点            3.00        1
#    2      电制氢出口阀后节点         2.98        1
#    3      储氢罐节点                29.80       1
#    4      燃料电池减压阀后节点       2.98        1
#    5      燃料电池入口节点           2.96        1
</HydroNode>

<HydroSource>
@    idx    name                       dev_type           node    control_type    pressure_set    flow_set    alpha    flow_min    flow_max    run_stat
#    1      交流电制氢-1_氢能设备端氢源    ac-electrolyzer    1       PRESSURE        3.00            0.0         1.0      0.0         2.0         1
</HydroSource>

<HydroLoad>
@    idx    name                         dev_type        node    flow_set    run_stat
#    1      直流燃料电池-1_氢能设备端氢荷    dc-fuel-cell    5       0.20        1
</HydroLoad>

<HydroValve>
@    idx    name                 dev_type                  i_node    j_node    control_type    conductance    flow_set    ratio    run_stat
#    1      电制氢出口阀门-1       hydrogen-valve           1         2         OPEN            3.0            0.0         1.0      1
#    2      燃料电池入口阀门-1     hydrogen-valve           4         5         OPEN            3.0            0.0         1.0      1
#    3      燃料电池入口减压阀-1   hydrogen-pressure-reducer 3         4         RATIO           1.0            0.0         0.1      1
</HydroValve>

<HydroCompressor>
@    idx    name                 dev_type              i_node    j_node    control_type    ratio    flow_set    run_stat
#    1      电制氢出口氢压机-1     hydrogen-compressor    2         3         RATIO           10.0     0.0         1
</HydroCompressor>

<HydroStorage>
@    idx    name              dev_type                   node    control_type    flow_set    alpha    flow_min    flow_max    run_stat
#    1      集装格式储氢罐-1    hydrogen-tank-container    3       FLOW            0.0         1.0      -2.0        2.0         1
</HydroStorage>

<AcE2Hydro>
@    idx    name          run_stat    idx_ac_load_t1    idx_h2_unit_t2
#    1      交流电制氢-1    1           2                 1
</AcE2Hydro>

<Hydro2DcE>
@    idx    name            run_stat    idx_dc_unit_t1    idx_h2_load_t2
#    1      直流燃料电池-1    1           10                1
</Hydro2DcE>

<ACWindGen>
@    idx    idx_acgenerator    wind_turbine_model    cut_in_wind_speed    rated_wind_speed    cut_out_wind_speed    rotor_diameter    hub_height
#    1      1                  WT-5MW                3                    12                  35                    6                 10
#    2      2                  WT-5MW                3                    12                  35                    6                 10
#    3      3                  WT-5MW                3                    12                  35                    6                 10
#    4      4                  WT-5MW                3                    12                  35                    6                 10
#    5      5                  WT-5MW                3                    12                  35                    6                 10
#    6      6                  WT-5MW                3                    12                  35                    6                 10
#    7      7                  WT-5MW                3                    12                  35                    6                 10
#    8      8                  WT-5MW                3                    12                  35                    6                 10
#    9      9                  WT-5MW                3                    12                  35                    6                 10
#    10     10                 WT-5MW                3                    12                  35                    6                 10
#    11     24                 WT-5MW                3                    12                  25                    170               110
</ACWindGen>

<DCPVGen>
@    idx    idx_dcgenerator    pv_module_model    module_efficiency    array_area    mppt_count    reference_irradiance    reference_temperature    temperature_coefficient
#    1      1                  Mono-550W          0.25                 200           25            1000                    25                       -0.004
#    2      2                  Mono-550W          0.25                 200           25            1000                    25                       -0.004
#    3      3                  Mono-550W          0.25                 200           25            1000                    25                       -0.004
</DCPVGen>

<DCStorageGen>
@    idx    idx_dcgenerator    storage_technology    battery_rack_count    energy_capacity    charge_discharge_efficiency    max_charge_power    max_discharge_power    state_of_charge    soc_upper_limit    soc_lower_limit
#    1      4                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
#    2      5                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
#    3      6                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
#    4      7                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
#    5      8                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
#    6      9                  lithium               20                    60                 0.95                           60                  60                     50                 90                 0.2
</DCStorageGen>

<ACStorageGen>
@    idx    idx_acgenerator    storage_technology    battery_rack_count    energy_capacity    charge_discharge_efficiency    max_charge_power    max_discharge_power    state_of_charge    soc_upper_limit    soc_lower_limit
#    1      23                 lithium               20                    200                90                             100                 100                    50                 90                 0.2
#    2      26                 lithium               20                    100                90                             100                 100                    50                 90                 0.2
</ACStorageGen>

<ACPVGen>
@    idx    idx_acgenerator    pv_module_model    module_efficiency    array_area    mppt_count    reference_irradiance    reference_temperature    temperature_coefficient
#    1      25                 Mono-550W          21.3                 100000        100           1000                    25                       -0.004
</ACPVGen>

<ACDieselGen>
@    idx    idx_acgenerator    diesel_unit_model    fuel_grade    specific_fuel_consumption    fuel_tank_capacity    rated_speed    start_time
#    1      27                 DG-2500              0#柴油         200                          20                    1500           10
#    2      28                 DG-2500              0#柴油         200                          20                    1500           10
#    3      29                 DG-2500              0#柴油         200                          20                    1500           10
#    4      30                 DG-2500              0#柴油         200                          20                    1500           10
</ACDieselGen>
