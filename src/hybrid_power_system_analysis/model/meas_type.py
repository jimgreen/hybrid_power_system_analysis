"""Integer codes for measurement device and measurement types."""

DEVICE_TYPE_ACNode = 1
DEVICE_TYPE_ACBranch = 2
DEVICE_TYPE_ACTransformer = 3
DEVICE_TYPE_ACLoad = 4
DEVICE_TYPE_ACGenerator = 5
DEVICE_TYPE_ACZeroBranch = 6
DEVICE_TYPE_ACZeroBranchConstraint = 7
DEVICE_TYPE_ACSwitchConstraint = 8
DEVICE_TYPE_ACSwitch = 9
DEVICE_TYPE_ACPowerBalance = 10
DEVICE_TYPE_DCNode = 11
DEVICE_TYPE_DCBranch = 12
DEVICE_TYPE_DCSwitch = 13
DEVICE_TYPE_DCZeroBranch = 14
DEVICE_TYPE_DCZeroBranchConstraint = 15
DEVICE_TYPE_DCSwitchConstraint = 16
DEVICE_TYPE_DCGenerator = 17
DEVICE_TYPE_DCLoad = 18
DEVICE_TYPE_DCDCConverter = 19
DEVICE_TYPE_DCACConverter = 20
DEVICE_TYPE_ACACConverter = 21
DEVICE_TYPE_ACBreak = 22
DEVICE_TYPE_ACBreakConstraint = 23
DEVICE_TYPE_DCBreak = 24
DEVICE_TYPE_DCBreakConstraint = 25
DEVICE_TYPE_ACThreeWindingTransformer = 26
DEVICE_TYPE_HeatNode = 27
DEVICE_TYPE_HeatPipe = 28
DEVICE_TYPE_HeatValve = 29
DEVICE_TYPE_HeatPump = 30
DEVICE_TYPE_HeatSource = 31
DEVICE_TYPE_HeatLoad = 32
DEVICE_TYPE_GasNode = 33
DEVICE_TYPE_GasPipe = 34
DEVICE_TYPE_GasValve = 35
DEVICE_TYPE_GasCompressor = 36
DEVICE_TYPE_GasSource = 37
DEVICE_TYPE_GasLoad = 38
DEVICE_TYPE_HydroNode = 39
DEVICE_TYPE_HydroPipe = 40
DEVICE_TYPE_HydroValve = 41
DEVICE_TYPE_HydroCompressor = 42
DEVICE_TYPE_HydroSource = 43
DEVICE_TYPE_HydroLoad = 44
DEVICE_TYPE_HeatExchanger = 45
DEVICE_TYPE_SteamNode = 46
DEVICE_TYPE_SteamPipe = 47
DEVICE_TYPE_SteamValve = 48
DEVICE_TYPE_SteamPressureReducer = 49
DEVICE_TYPE_SteamSource = 50
DEVICE_TYPE_SteamLoad = 51

DEVICE_TYPE_CODES = {
    "ACNode": DEVICE_TYPE_ACNode,
    "ACBranch": DEVICE_TYPE_ACBranch,
    "ACTransformer": DEVICE_TYPE_ACTransformer,
    "ACLoad": DEVICE_TYPE_ACLoad,
    "ACGenerator": DEVICE_TYPE_ACGenerator,
    "ACZeroBranch": DEVICE_TYPE_ACZeroBranch,
    "ACZeroBranchConstraint": DEVICE_TYPE_ACZeroBranchConstraint,
    "ACSwitchConstraint": DEVICE_TYPE_ACSwitchConstraint,
    "ACSwitch": DEVICE_TYPE_ACSwitch,
    "ACPowerBalance": DEVICE_TYPE_ACPowerBalance,
    "DCNode": DEVICE_TYPE_DCNode,
    "DCBranch": DEVICE_TYPE_DCBranch,
    "DCSwitch": DEVICE_TYPE_DCSwitch,
    "DCZeroBranch": DEVICE_TYPE_DCZeroBranch,
    "DCZeroBranchConstraint": DEVICE_TYPE_DCZeroBranchConstraint,
    "DCSwitchConstraint": DEVICE_TYPE_DCSwitchConstraint,
    "DCGenerator": DEVICE_TYPE_DCGenerator,
    "DCLoad": DEVICE_TYPE_DCLoad,
    "DCDCConverter": DEVICE_TYPE_DCDCConverter,
    "DCACConverter": DEVICE_TYPE_DCACConverter,
    "ACACConverter": DEVICE_TYPE_ACACConverter,
    "ACBreak": DEVICE_TYPE_ACBreak,
    "ACBreakConstraint": DEVICE_TYPE_ACBreakConstraint,
    "DCBreak": DEVICE_TYPE_DCBreak,
    "DCBreakConstraint": DEVICE_TYPE_DCBreakConstraint,
    "AC3WTransformer": DEVICE_TYPE_ACThreeWindingTransformer,
    "ACThreeWindingTransformer": DEVICE_TYPE_ACThreeWindingTransformer,
    "HeatNode": DEVICE_TYPE_HeatNode,
    "HeatPipe": DEVICE_TYPE_HeatPipe,
    "HeatValve": DEVICE_TYPE_HeatValve,
    "HeatPump": DEVICE_TYPE_HeatPump,
    "HeatSource": DEVICE_TYPE_HeatSource,
    "HeatLoad": DEVICE_TYPE_HeatLoad,
    "GasNode": DEVICE_TYPE_GasNode,
    "GasPipe": DEVICE_TYPE_GasPipe,
    "GasValve": DEVICE_TYPE_GasValve,
    "GasCompressor": DEVICE_TYPE_GasCompressor,
    "GasSource": DEVICE_TYPE_GasSource,
    "GasLoad": DEVICE_TYPE_GasLoad,
    "HydroNode": DEVICE_TYPE_HydroNode,
    "HydroPipe": DEVICE_TYPE_HydroPipe,
    "HydroValve": DEVICE_TYPE_HydroValve,
    "HydroCompressor": DEVICE_TYPE_HydroCompressor,
    "HydroSource": DEVICE_TYPE_HydroSource,
    "HydroLoad": DEVICE_TYPE_HydroLoad,
    "HeatExchanger": DEVICE_TYPE_HeatExchanger,
    "SteamNode": DEVICE_TYPE_SteamNode,
    "SteamPipe": DEVICE_TYPE_SteamPipe,
    "SteamValve": DEVICE_TYPE_SteamValve,
    "SteamPressureReducer": DEVICE_TYPE_SteamPressureReducer,
    "SteamSource": DEVICE_TYPE_SteamSource,
    "SteamLoad": DEVICE_TYPE_SteamLoad,
}
DEVICE_TYPE_NAMES = {code: name for name, code in DEVICE_TYPE_CODES.items()}

MEAS_TYPE_UNKNOWN = 0
MEAS_TYPE_V = 1
MEAS_TYPE_ANGLE = 2
MEAS_TYPE_THETA = 3
MEAS_TYPE_P_FROM = 4
MEAS_TYPE_Q_FROM = 5
MEAS_TYPE_V_FROM = 6
MEAS_TYPE_I_FROM = 7
MEAS_TYPE_P_TO = 8
MEAS_TYPE_Q_TO = 9
MEAS_TYPE_V_TO = 10
MEAS_TYPE_I_TO = 11
MEAS_TYPE_P_LOAD = 12
MEAS_TYPE_Q_LOAD = 13
MEAS_TYPE_V_LOAD = 14
MEAS_TYPE_I_LOAD = 15
MEAS_TYPE_P_GEN = 16
MEAS_TYPE_Q_GEN = 17
MEAS_TYPE_V_GEN = 18
MEAS_TYPE_I_GEN = 19
MEAS_TYPE_P_BALANCE = 20
MEAS_TYPE_Q_BALANCE = 21
MEAS_TYPE_V_DIFF = 22
MEAS_TYPE_ANGLE_DIFF = 23
MEAS_TYPE_THETA_DIFF = 24
MEAS_TYPE_P_DC = 25
MEAS_TYPE_V_DC = 26
MEAS_TYPE_I_DC = 27
MEAS_TYPE_P_AC = 28
MEAS_TYPE_Q_AC = 29
MEAS_TYPE_V_AC = 30
MEAS_TYPE_I_AC = 31
MEAS_TYPE_P_IN = 32
MEAS_TYPE_P_OUT = 33
MEAS_TYPE_I_OUT = 34
MEAS_TYPE_P_THIRD = 35
MEAS_TYPE_Q_THIRD = 36
MEAS_TYPE_V_THIRD = 37
MEAS_TYPE_I_THIRD = 38
MEAS_TYPE_PRESSURE = 39
MEAS_TYPE_FLOW_FROM = 40
MEAS_TYPE_FLOW_TO = 41
MEAS_TYPE_FLOW = 42
MEAS_TYPE_T_SUPPLY = 43
MEAS_TYPE_T_RETURN = 44
MEAS_TYPE_HEAT = 45
MEAS_TYPE_PRESSURE_FROM = 46
MEAS_TYPE_PRESSURE_TO = 47
MEAS_TYPE_TS_FROM = 48
MEAS_TYPE_TS_TO = 49
MEAS_TYPE_TR_FROM = 50
MEAS_TYPE_TR_TO = 51
MEAS_TYPE_ENTHALPY = 52
MEAS_TYPE_TEMPERATURE = 53
MEAS_TYPE_H_FROM = 54
MEAS_TYPE_H_TO = 55
MEAS_TYPE_T_FROM = 56
MEAS_TYPE_T_TO = 57

MEAS_TYPE_CODES = {
    "UNKNOWN": MEAS_TYPE_UNKNOWN,
    "V": MEAS_TYPE_V,
    "ANGLE": MEAS_TYPE_ANGLE,
    "THETA": MEAS_TYPE_THETA,
    "P_FROM": MEAS_TYPE_P_FROM,
    "Q_FROM": MEAS_TYPE_Q_FROM,
    "V_FROM": MEAS_TYPE_V_FROM,
    "I_FROM": MEAS_TYPE_I_FROM,
    "P_TO": MEAS_TYPE_P_TO,
    "Q_TO": MEAS_TYPE_Q_TO,
    "V_TO": MEAS_TYPE_V_TO,
    "I_TO": MEAS_TYPE_I_TO,
    "P_LOAD": MEAS_TYPE_P_LOAD,
    "Q_LOAD": MEAS_TYPE_Q_LOAD,
    "V_LOAD": MEAS_TYPE_V_LOAD,
    "I_LOAD": MEAS_TYPE_I_LOAD,
    "P_GEN": MEAS_TYPE_P_GEN,
    "Q_GEN": MEAS_TYPE_Q_GEN,
    "V_GEN": MEAS_TYPE_V_GEN,
    "I_GEN": MEAS_TYPE_I_GEN,
    "P_BALANCE": MEAS_TYPE_P_BALANCE,
    "Q_BALANCE": MEAS_TYPE_Q_BALANCE,
    "V_DIFF": MEAS_TYPE_V_DIFF,
    "ANGLE_DIFF": MEAS_TYPE_ANGLE_DIFF,
    "THETA_DIFF": MEAS_TYPE_THETA_DIFF,
    "P_DC": MEAS_TYPE_P_DC,
    "V_DC": MEAS_TYPE_V_DC,
    "I_DC": MEAS_TYPE_I_DC,
    "P_AC": MEAS_TYPE_P_AC,
    "Q_AC": MEAS_TYPE_Q_AC,
    "V_AC": MEAS_TYPE_V_AC,
    "I_AC": MEAS_TYPE_I_AC,
    "P_IN": MEAS_TYPE_P_IN,
    "P_OUT": MEAS_TYPE_P_OUT,
    "I_OUT": MEAS_TYPE_I_OUT,
    "P_THIRD": MEAS_TYPE_P_THIRD,
    "Q_THIRD": MEAS_TYPE_Q_THIRD,
    "V_THIRD": MEAS_TYPE_V_THIRD,
    "I_THIRD": MEAS_TYPE_I_THIRD,
    "PRESSURE": MEAS_TYPE_PRESSURE,
    "FLOW_FROM": MEAS_TYPE_FLOW_FROM,
    "FLOW_TO": MEAS_TYPE_FLOW_TO,
    "FLOW": MEAS_TYPE_FLOW,
    "T_SUPPLY": MEAS_TYPE_T_SUPPLY,
    "T_RETURN": MEAS_TYPE_T_RETURN,
    "HEAT": MEAS_TYPE_HEAT,
    "PRESSURE_FROM": MEAS_TYPE_PRESSURE_FROM,
    "PRESSURE_TO": MEAS_TYPE_PRESSURE_TO,
    "TS_FROM": MEAS_TYPE_TS_FROM,
    "TS_TO": MEAS_TYPE_TS_TO,
    "TR_FROM": MEAS_TYPE_TR_FROM,
    "TR_TO": MEAS_TYPE_TR_TO,
    "ENTHALPY": MEAS_TYPE_ENTHALPY,
    "TEMPERATURE": MEAS_TYPE_TEMPERATURE,
    "H_FROM": MEAS_TYPE_H_FROM,
    "H_TO": MEAS_TYPE_H_TO,
    "T_FROM": MEAS_TYPE_T_FROM,
    "T_TO": MEAS_TYPE_T_TO,
}
MEAS_TYPE_NAMES = {code: name for name, code in MEAS_TYPE_CODES.items()}
MEAS_TYPE_CODES.update(
    {
        "P_K": MEAS_TYPE_P_THIRD,
        "Q_K": MEAS_TYPE_Q_THIRD,
        "V_K": MEAS_TYPE_V_THIRD,
        "I_K": MEAS_TYPE_I_THIRD,
    }
)
