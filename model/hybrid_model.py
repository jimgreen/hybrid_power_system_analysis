from  ac_model import  *
from  dc_model import  *
class DCACConverter:
    def __init__(
            self, idx, ac_node, dc_node, r1, r2, control_type,
            p_ac_set, q_ac_set, v_ac_set, v_dc_set, run_stat=1):
        self.idx = idx
        self.ac_node = ac_node
        self.dc_node = dc_node
        self.r1 = r1
        self.r2 = r2
        self.control_type = control_type
        self.p_ac_set = p_ac_set
        self.q_ac_set = q_ac_set
        self.v_ac_set = v_ac_set
        self.v_dc_set = v_dc_set
        self.run_stat = run_stat
        self.dc_p = None
        self.ac_p = None
        self.ac_q = None
        self.dc_i = None
        self.ac_i = None
        self.ac_node_obj = None
        self.dc_node_obj = None


class ACACConverter:
    def __init__(
            self, idx, i_node, j_node, r1, r2, control_type,
            p_set, i_q_set, j_q_set, i_v_set, j_v_set, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r1 = r1
        self.r2 = r2
        self.control_type = control_type
        self.p_set = p_set
        self.i_q_set = i_q_set
        self.j_q_set = j_q_set
        self.i_v_set = i_v_set
        self.j_v_set = j_v_set
        self.run_stat = run_stat
        self.i_p = None
        self.i_q = None
        self.j_p = None
        self.j_q = None
        self.i_i = None
        self.j_i = None
        self.i_node_obj = None
        self.j_node_obj = None
