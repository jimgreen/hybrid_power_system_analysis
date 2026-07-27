import math
from dataclasses import dataclass


MODEL_BLOCK = "Model"
POWER_BASE_BLOCK = "PowerBase"


@dataclass(frozen=True)
class UnitSettings:
    p_base: float
    u_scale: float
    p_scale: float
    i_scale: float
    p_unit: str = "kW"
    u_unit: str = "kV"
    i_unit: str = "kA"

    @property
    def p_base_kW(self) -> float:
        return self.p_base / self.p_scale


def _positive_float(obj, attr: str, block_name: str) -> float:
    if not hasattr(obj, attr):
        raise RuntimeError(
            f"E file <{block_name}> must define p_base, u_unit, p_unit, and i_unit"
        )
    value = float(getattr(obj, attr))
    if value <= 0.0:
        raise RuntimeError(f"Invalid {attr} in <{block_name}>: {value}")
    return value


_P_UNIT_TO_SCALE = {
    "W": 1000.0,
    "kW": 1.0,
    "MW": 0.001,
}
_U_UNIT_TO_SCALE = {
    "mV": 1_000_000.0,
    "V": 1000.0,
    "kV": 1.0,
}
_I_UNIT_TO_SCALE = {
    "mA": 1_000_000.0,
    "A": 1000.0,
    "kA": 1.0,
}


def _unit_scale(
    obj,
    unit_attr: str,
    scale_attr: str,
    mapping,
    default_unit: str,
    block_name: str,
) -> tuple[float, str]:
    """Return the numeric scale implied by the named E-file unit.

    New E files define p/u/i units by name.  Legacy in-memory test objects may
    still carry *_scale; keep accepting those objects while normalizing all
    runtime models to the same numeric scale fields.
    """
    unit_value = getattr(obj, unit_attr, None)
    if unit_value not in (None, ""):
        unit = str(unit_value).strip()
        if unit not in mapping:
            allowed = "/".join(mapping.keys())
            raise RuntimeError(f"Invalid {unit_attr} in <{block_name}>: {unit!r}; expected one of {allowed}")
        return float(mapping[unit]), unit
    return _positive_float(obj, scale_attr, block_name), default_unit


def get_unit_settings(model) -> UnitSettings:
    block_name = MODEL_BLOCK
    rows = getattr(model, block_name, None)
    if not rows:
        block_name = POWER_BASE_BLOCK
        rows = getattr(model, block_name, None)
    if not rows:
        raise RuntimeError(
            "E file must define <Model> with p_base, u_unit, p_unit, and i_unit"
        )
    row = rows[0]
    u_scale, u_unit = _unit_scale(row, "u_unit", "u_scale", _U_UNIT_TO_SCALE, "kV", block_name)
    p_scale, p_unit = _unit_scale(row, "p_unit", "p_scale", _P_UNIT_TO_SCALE, "kW", block_name)
    i_scale, i_unit = _unit_scale(row, "i_unit", "i_scale", _I_UNIT_TO_SCALE, "kA", block_name)
    return UnitSettings(
        p_base=_positive_float(row, "p_base", block_name),
        u_scale=u_scale,
        p_scale=p_scale,
        i_scale=i_scale,
        p_unit=p_unit,
        u_unit=u_unit,
        i_unit=i_unit,
    )


def ac_current_base_ka(power_base_kW: float, vbase_kv: float) -> float:
    if abs(vbase_kv) <= 1e-12:
        return 1.0
    return power_base_kW / (1000.0 * math.sqrt(3.0) * abs(vbase_kv))


def dc_current_base_ka(power_base_kW: float, vbase_kv: float) -> float:
    if abs(vbase_kv) <= 1e-12:
        return 1.0
    return power_base_kW / (1000.0 * abs(vbase_kv))


def _scale_attr(obj, attr: str, scale: float) -> None:
    value = getattr(obj, attr, None)
    if value is None:
        return
    setattr(obj, attr, float(value) / scale)


def _scale_attr_in_dict(obj, attr: str, scale: float) -> None:
    values = obj.__dict__
    value = values.get(attr)
    if value is not None:
        values[attr] = float(value) / scale


def _scale_power_attrs_in_dict(obj, attrs, p_base: float) -> None:
    values = obj.__dict__
    for attr in attrs:
        value = values.get(attr)
        if value is not None:
            values[attr] = float(value) / p_base


def _any_attr_value(devices, attrs) -> bool:
    for obj in devices:
        values = getattr(obj, "__dict__", {})
        for attr in attrs:
            if values.get(attr) is not None:
                return True
    return False


def _scale_voltage_attr(obj, attr: str, node, settings: UnitSettings) -> None:
    if node is None:
        return
    _scale_attr(obj, attr, settings.u_scale * float(node.vbase))

def _angle_deg_to_rad(obj, attr: str) -> None:
    value = getattr(obj, attr, None)
    if value is None:
        return
    setattr(obj, attr, math.radians(float(value)))


def normalize_model_named_units(model) -> float:
    if getattr(model, "_named_units_normalized", False):
        return float(model.p_base)

    settings = get_unit_settings(model)
    setattr(model, "p_base", settings.p_base)
    setattr(model, "p_base_kW", settings.p_base_kW)
    setattr(model, "u_scale", settings.u_scale)
    setattr(model, "p_scale", settings.p_scale)
    setattr(model, "i_scale", settings.i_scale)
    setattr(model, "p_unit", settings.p_unit)
    setattr(model, "u_unit", settings.u_unit)
    setattr(model, "i_unit", settings.i_unit)

    ac_nodes = {node.idx: node for node in getattr(model, "ACNode", [])}
    dc_nodes = {node.idx: node for node in getattr(model, "DCNode", [])}
    p_base = settings.p_base
    u_scale = settings.u_scale
    p_base_kW = settings.p_base_kW
    i_scale = settings.i_scale

    for node in ac_nodes.values():
        _scale_attr_in_dict(node, "vbase", u_scale)
        _scale_voltage_attr(node, "voltage", node, settings)
        _angle_deg_to_rad(node, "angle")
    ac_current_scales = {
        node_idx: i_scale * ac_current_base_ka(p_base_kW, float(node.vbase))
        for node_idx, node in ac_nodes.items()
    }

    ac_branches = getattr(model, "ACBranch", [])
    if _any_attr_value(ac_branches, ("i_p", "i_q", "j_p", "j_q", "i_c", "j_c")):
        for br in ac_branches:
            _scale_power_attrs_in_dict(br, ("i_p", "i_q", "j_p", "j_q"), p_base)
            i_scale_base = ac_current_scales.get(br.i_node)
            if i_scale_base is not None:
                _scale_attr_in_dict(br, "i_c", i_scale_base)
            j_scale_base = ac_current_scales.get(br.j_node)
            if j_scale_base is not None:
                _scale_attr_in_dict(br, "j_c", j_scale_base)

    for tr in getattr(model, "ACTransformer", []):
        if not hasattr(tr, "gt"):
            tr.gt = 0.0
        if not hasattr(tr, "bt") and hasattr(tr, "b"):
            tr.bt = float(tr.b) / 2.0
        _scale_power_attrs_in_dict(tr, ("i_p", "i_q", "j_p", "j_q"), p_base)
        i_scale_base = ac_current_scales.get(tr.i_node)
        if i_scale_base is not None:
            _scale_attr_in_dict(tr, "i_c", i_scale_base)
        j_scale_base = ac_current_scales.get(tr.j_node)
        if j_scale_base is not None:
            _scale_attr_in_dict(tr, "j_c", j_scale_base)

    three_winding_transformers = getattr(model, "ACThreeWindingTransformer", None)
    if three_winding_transformers is None:
        three_winding_transformers = getattr(model, "AC3WTransformer", [])
    for tr in three_winding_transformers:
        _scale_power_attrs_in_dict(
            tr,
            ("i_p", "i_q", "j_p", "j_q", "k_p", "k_q"),
            p_base,
        )
        for terminal in ("i", "j", "k"):
            scale_base = ac_current_scales.get(getattr(tr, f"{terminal}_node", None))
            if scale_base is not None:
                _scale_attr_in_dict(tr, f"{terminal}_c", scale_base)

    for load in getattr(model, "ACLoad", []):
        if not hasattr(load, "pbase"):
            load.pbase = 1.0
        if not hasattr(load, "qbase"):
            load.qbase = 1.0
        _scale_power_attrs_in_dict(load, ("pbase", "qbase", "p", "q"), p_base)
        scale_base = ac_current_scales.get(load.node)
        if scale_base is not None:
            _scale_attr_in_dict(load, "current", scale_base)

    for gen in getattr(model, "ACGenerator", []):
        node = ac_nodes.get(gen.node)
        _scale_power_attrs_in_dict(gen, ("p_set", "q_set", "p_max", "p", "q"), p_base)
        _scale_voltage_attr(gen, "v_set", node, settings)
        scale_base = ac_current_scales.get(gen.node)
        if scale_base is not None:
            _scale_attr_in_dict(gen, "current", scale_base)

    for sc in getattr(model, "ACShuntCompensator", []):
        node = ac_nodes.get(sc.node)
        _scale_power_attrs_in_dict(sc, ("q_set", "p", "q"), p_base)
        _scale_voltage_attr(sc, "v_set", node, settings)
        scale_base = ac_current_scales.get(sc.node)
        if scale_base is not None:
            _scale_attr_in_dict(sc, "current", scale_base)

    ac_zero_switches = [*getattr(model, "ACZeroBranch", []), *getattr(model, "ACSwitch", [])]
    if _any_attr_value(ac_zero_switches, ("p", "q", "current")):
        for dev in ac_zero_switches:
            _scale_power_attrs_in_dict(dev, ("p", "q"), p_base)
            scale_base = ac_current_scales.get(dev.i_node)
            if scale_base is not None:
                _scale_attr_in_dict(dev, "current", scale_base)

    for node in dc_nodes.values():
        _scale_attr_in_dict(node, "vbase", u_scale)
        _scale_voltage_attr(node, "voltage", node, settings)
    dc_current_scales = {
        node_idx: i_scale * dc_current_base_ka(p_base_kW, float(node.vbase))
        for node_idx, node in dc_nodes.items()
    }

    dc_branches = getattr(model, "DCBranch", [])
    if _any_attr_value(dc_branches, ("i_p", "j_p", "current")):
        for br in dc_branches:
            _scale_power_attrs_in_dict(br, ("i_p", "j_p"), p_base)
            scale_base = dc_current_scales.get(br.i_node)
            if scale_base is not None:
                _scale_attr_in_dict(br, "current", scale_base)

    for load in getattr(model, "DCLoad", []):
        if not hasattr(load, "pbase"):
            load.pbase = 1.0
        _scale_power_attrs_in_dict(load, ("pbase", "p"), p_base)
        scale_base = dc_current_scales.get(load.node)
        if scale_base is not None:
            _scale_attr_in_dict(load, "current", scale_base)

    for gen in getattr(model, "DCGenerator", []):
        node = dc_nodes.get(gen.node)
        _scale_power_attrs_in_dict(gen, ("p_set", "p"), p_base)
        _scale_voltage_attr(gen, "v_set", node, settings)
        scale_base = dc_current_scales.get(gen.node)
        if scale_base is not None:
            _scale_attr_in_dict(gen, "i_set", scale_base)
            _scale_attr_in_dict(gen, "current", scale_base)

    dc_zero_switches = [*getattr(model, "DCZeroBranch", []), *getattr(model, "DCSwitch", [])]
    if _any_attr_value(dc_zero_switches, ("p", "current")):
        for dev in dc_zero_switches:
            _scale_power_attrs_in_dict(dev, ("p",), p_base)
            scale_base = dc_current_scales.get(dev.i_node)
            if scale_base is not None:
                _scale_attr_in_dict(dev, "current", scale_base)

    for conv in getattr(model, "DCDCConverter", []):
        i_node = dc_nodes.get(conv.i_node)
        j_node = dc_nodes.get(conv.j_node)
        _scale_power_attrs_in_dict(conv, ("p_set", "i_p", "j_p"), p_base)
        i_ctrl = str(getattr(conv, "i_control_type", "") or "").upper()
        j_ctrl = str(getattr(conv, "j_control_type", "") or "").upper()
        if not i_ctrl and not j_ctrl:
            i_ctrl = str(getattr(conv, "control_type", "P") or "P").upper()
            j_ctrl = "SLACK"
        elif not i_ctrl:
            i_ctrl = "SLACK"
        elif not j_ctrl:
            j_ctrl = "SLACK"
        v_node = j_node if j_ctrl in ("V", "CTRL_V") and i_ctrl in ("SLACK", "CTRL_SLACK") else i_node
        _scale_voltage_attr(conv, "v_set", v_node, settings)
        i_scale_base = dc_current_scales.get(conv.i_node)
        if i_scale_base is not None:
            _scale_attr_in_dict(conv, "i_set", i_scale_base)
            _scale_attr_in_dict(conv, "i_c", i_scale_base)
        j_scale_base = dc_current_scales.get(conv.j_node)
        if j_scale_base is not None:
            _scale_attr_in_dict(conv, "j_c", j_scale_base)

    for conv in getattr(model, "DCACConverter", []):
        ac_node = ac_nodes.get(conv.ac_node)
        dc_node = dc_nodes.get(conv.dc_node)
        _scale_power_attrs_in_dict(conv, ("p_ac_set", "q_ac_set", "dc_p", "ac_p", "ac_q"), p_base)
        _scale_voltage_attr(conv, "v_ac_set", ac_node, settings)
        _scale_voltage_attr(conv, "v_dc_set", dc_node, settings)
        dc_scale_base = dc_current_scales.get(conv.dc_node)
        if dc_scale_base is not None:
            _scale_attr_in_dict(conv, "dc_i", dc_scale_base)
        ac_scale_base = ac_current_scales.get(conv.ac_node)
        if ac_scale_base is not None:
            _scale_attr_in_dict(conv, "ac_i", ac_scale_base)

    for conv in getattr(model, "ACACConverter", []):
        i_node = ac_nodes.get(conv.i_node)
        j_node = ac_nodes.get(conv.j_node)
        _scale_power_attrs_in_dict(conv, ("p_set", "i_q_set", "j_q_set", "i_p", "i_q", "j_p", "j_q"), p_base)
        _scale_voltage_attr(conv, "i_v_set", i_node, settings)
        _scale_voltage_attr(conv, "j_v_set", j_node, settings)
        i_scale_base = ac_current_scales.get(conv.i_node)
        if i_scale_base is not None:
            _scale_attr_in_dict(conv, "i_i", i_scale_base)
        j_scale_base = ac_current_scales.get(conv.j_node)
        if j_scale_base is not None:
            _scale_attr_in_dict(conv, "j_i", j_scale_base)

    setattr(model, "_named_units_normalized", True)
    return settings.p_base
