import math
from dataclasses import dataclass


POWER_BASE_BLOCK = "PowerBase"


@dataclass(frozen=True)
class UnitSettings:
    p_base: float
    u_scale: float
    p_scale: float
    i_scale: float

    @property
    def p_base_kW(self) -> float:
        return self.p_base / self.p_scale


def _positive_float(obj, attr: str) -> float:
    if not hasattr(obj, attr):
        raise RuntimeError(
            "E file <PowerBase> must define p_base, u_scale, p_scale, and i_scale"
        )
    value = float(getattr(obj, attr))
    if value <= 0.0:
        raise RuntimeError(f"Invalid {attr} in <PowerBase>: {value}")
    return value


def get_unit_settings(model) -> UnitSettings:
    rows = getattr(model, POWER_BASE_BLOCK, None)
    if not rows:
        raise RuntimeError(
            "E file must define <PowerBase> with p_base, u_scale, p_scale, and i_scale"
        )
    row = rows[0]
    return UnitSettings(
        p_base=_positive_float(row, "p_base"),
        u_scale=_positive_float(row, "u_scale"),
        p_scale=_positive_float(row, "p_scale"),
        i_scale=_positive_float(row, "i_scale"),
    )


def get_power_base_kw(model) -> float:
    return get_unit_settings(model).p_base_kW


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


def _scale_power_attrs(obj, attrs, settings: UnitSettings) -> None:
    for attr in attrs:
        _scale_attr(obj, attr, settings.p_base)


def _scale_voltage_attr(obj, attr: str, node, settings: UnitSettings) -> None:
    if node is None:
        return
    _scale_attr(obj, attr, settings.u_scale * float(node.vbase))


def _scale_current_attr(obj, attr: str, node, settings: UnitSettings, is_ac: bool) -> None:
    if node is None:
        return
    base = (
        ac_current_base_ka(settings.p_base_kW, float(node.vbase))
        if is_ac
        else dc_current_base_ka(settings.p_base_kW, float(node.vbase))
    )
    _scale_attr(obj, attr, settings.i_scale * base)


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

    ac_nodes = {node.idx: node for node in getattr(model, "ACNode", [])}
    dc_nodes = {node.idx: node for node in getattr(model, "DCNode", [])}

    for node in ac_nodes.values():
        _scale_attr(node, "vbase", settings.u_scale)
        _scale_voltage_attr(node, "voltage", node, settings)
        _angle_deg_to_rad(node, "angle")

    for br in getattr(model, "ACBranch", []):
        i_node = ac_nodes.get(br.i_node)
        j_node = ac_nodes.get(br.j_node)
        _scale_power_attrs(br, ("i_p", "i_q", "j_p", "j_q"), settings)
        _scale_current_attr(br, "i_c", i_node, settings, True)
        _scale_current_attr(br, "j_c", j_node, settings, True)

    for tr in getattr(model, "ACTransformer", []):
        i_node = ac_nodes.get(tr.i_node)
        j_node = ac_nodes.get(tr.j_node)
        _scale_power_attrs(tr, ("i_p", "i_q", "j_p", "j_q"), settings)
        _scale_current_attr(tr, "i_c", i_node, settings, True)
        _scale_current_attr(tr, "j_c", j_node, settings, True)

    for load in getattr(model, "ACLoad", []):
        node = ac_nodes.get(load.node)
        _scale_power_attrs(load, ("pv0", "pv1", "pv2", "qv0", "qv1", "qv2", "p", "q"), settings)
        _scale_current_attr(load, "current", node, settings, True)

    for gen in getattr(model, "ACGenerator", []):
        node = ac_nodes.get(gen.node)
        _scale_power_attrs(gen, ("p_set", "q_set", "p", "q"), settings)
        _scale_voltage_attr(gen, "v_set", node, settings)
        _scale_current_attr(gen, "current", node, settings, True)

    for sc in getattr(model, "ACShuntCompensator", []):
        node = ac_nodes.get(sc.node)
        _scale_power_attrs(sc, ("q_set", "p", "q"), settings)
        _scale_voltage_attr(sc, "v_set", node, settings)
        _scale_current_attr(sc, "current", node, settings, True)

    for dev in [*getattr(model, "ACZeroBranch", []), *getattr(model, "ACSwitch", [])]:
        i_node = ac_nodes.get(dev.i_node)
        _scale_power_attrs(dev, ("p", "q"), settings)
        _scale_current_attr(dev, "current", i_node, settings, True)

    for node in dc_nodes.values():
        _scale_attr(node, "vbase", settings.u_scale)
        _scale_voltage_attr(node, "voltage", node, settings)

    for br in getattr(model, "DCBranch", []):
        i_node = dc_nodes.get(br.i_node)
        _scale_power_attrs(br, ("i_p", "j_p"), settings)
        _scale_current_attr(br, "current", i_node, settings, False)

    for load in getattr(model, "DCLoad", []):
        node = dc_nodes.get(load.node)
        _scale_power_attrs(load, ("pv0", "pv1", "pv2", "p"), settings)
        _scale_current_attr(load, "current", node, settings, False)

    for gen in getattr(model, "DCGenerator", []):
        node = dc_nodes.get(gen.node)
        _scale_power_attrs(gen, ("p_set", "p"), settings)
        _scale_voltage_attr(gen, "v_set", node, settings)
        _scale_current_attr(gen, "i_set", node, settings, False)
        _scale_current_attr(gen, "current", node, settings, False)

    for dev in [*getattr(model, "DCZeroBranch", []), *getattr(model, "DCSwitch", [])]:
        i_node = dc_nodes.get(dev.i_node)
        _scale_power_attrs(dev, ("p",), settings)
        _scale_current_attr(dev, "current", i_node, settings, False)

    for conv in getattr(model, "DCDCConverter", []):
        i_node = dc_nodes.get(conv.i_node)
        j_node = dc_nodes.get(conv.j_node)
        _scale_power_attrs(conv, ("p_set", "i_p", "j_p"), settings)
        _scale_voltage_attr(conv, "v_set", i_node, settings)
        _scale_current_attr(conv, "i_set", i_node, settings, False)
        _scale_current_attr(conv, "i_c", i_node, settings, False)
        _scale_current_attr(conv, "j_c", j_node, settings, False)

    for conv in getattr(model, "DCACConverter", []):
        ac_node = ac_nodes.get(conv.ac_node)
        dc_node = dc_nodes.get(conv.dc_node)
        _scale_power_attrs(conv, ("p_ac_set", "q_ac_set", "dc_p", "ac_p", "ac_q"), settings)
        _scale_voltage_attr(conv, "v_ac_set", ac_node, settings)
        _scale_voltage_attr(conv, "v_dc_set", dc_node, settings)
        _scale_current_attr(conv, "dc_i", dc_node, settings, False)
        _scale_current_attr(conv, "ac_i", ac_node, settings, True)

    for conv in getattr(model, "ACACConverter", []):
        i_node = ac_nodes.get(conv.i_node)
        j_node = ac_nodes.get(conv.j_node)
        _scale_power_attrs(conv, ("p_set", "i_q_set", "j_q_set", "i_p", "i_q", "j_p", "j_q"), settings)
        _scale_voltage_attr(conv, "i_v_set", i_node, settings)
        _scale_voltage_attr(conv, "j_v_set", j_node, settings)
        _scale_current_attr(conv, "i_i", i_node, settings, True)
        _scale_current_attr(conv, "j_i", j_node, settings, True)

    setattr(model, "_named_units_normalized", True)
    return settings.p_base
