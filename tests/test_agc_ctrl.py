import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

from agc import agc_ctrl


class AGCControlTest(unittest.TestCase):
    def test_agc_control_once_runs_one_control_cycle_without_gather(self):
        output = Path("tmp_test/agc_once_dev_ctrl.e")
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            output.unlink()
        self.addCleanup(lambda: output.exists() and output.unlink())

        model = SimpleNamespace(
            wind_generator=[],
            pv_generator=[],
            estorage=[],
            diesel_generator=[],
            yt=[],
        )
        config = SimpleNamespace(
            model=model,
            logger=logging.getLogger("test"),
            gather_count=3,
            control_count=2,
            ctrl_output_file=output,
        )
        calls = []

        def control(logger, emodel):
            calls.append("control")
            emodel.agc_balance_mismatch = 0.0

        def check(logger, emodel):
            calls.append("check")

        agc_ctrl.agc_control_once(config, control, check)

        self.assertEqual(config.gather_count, 3)
        self.assertEqual(config.control_count, 3)
        self.assertEqual(calls, ["control", "check"])
        self.assertTrue(output.exists())

    def test_agc_control_step_runs_one_gather_and_control_cycle(self):
        output = Path("tmp_test/agc_step_dev_ctrl.e")
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            output.unlink()
        self.addCleanup(lambda: output.exists() and output.unlink())

        model = SimpleNamespace(
            yc=[],
            yx=[],
            diesel_generator=[],
            pv_generator=[],
            wind_generator=[],
            energyconsumer=[],
            estorage=[],
            yt=[],
            wind_speed=None,
            solor_irrad=None,
            env_temp=None,
        )
        config = SimpleNamespace(
            model=model,
            logger=logging.getLogger("test"),
            gather_count=0,
            control_count=0,
            ctrl_output_file=output,
            fes_client=SimpleNamespace(
                get_yc_value=lambda schema, context, rtu, pnt: (False, "", None),
                get_yx_value=lambda schema, context, rtu, pnt: (False, "", None),
                schema=None,
                context=None,
            ),
        )
        calls = []

        def forecast(logger, emodel, wind_speed, solar_irrad, env_temp):
            calls.append("forecast")

        def control(logger, emodel):
            calls.append("control")
            emodel.agc_balance_mismatch = 0.0

        def check(logger, emodel):
            calls.append("check")

        agc_ctrl.agc_control_step(config, control, check, forecast)

        self.assertEqual(config.gather_count, 1)
        self.assertEqual(config.control_count, 1)
        self.assertEqual(calls, ["forecast", "control", "check"])
        self.assertTrue(output.exists())

    def test_export_control_results_writes_dev_ctrl_e(self):
        output = Path("tmp_test/agc_unit_dev_ctrl.e")
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            output.unlink()
        self.addCleanup(lambda: output.exists() and output.unlink())

        model = SimpleNamespace(
            wind_generator=[
                SimpleNamespace(id=1, name="wind", p_ctrl=12.5, p_cur=11.5),
            ],
            pv_generator=[
                SimpleNamespace(id=2, name="pv", p_ctrl=7.5, p_cur=7.0),
            ],
            estorage=[
                SimpleNamespace(id=3, name="storage", p_ctrl=-4.0, p_cur=-3.5),
            ],
            diesel_generator=[
                SimpleNamespace(id=4, name="diesel", p_ctrl=80.0, p_cur=79.5),
            ],
            yt=[
                SimpleNamespace(id=5, name="hydrogen", rtu=3, pnt=3, ctrl_value=6.0),
            ],
            agc_balance_mismatch=1.25,
        )

        agc_ctrl.export_control_results(model, output)

        text = output.read_text(encoding="utf-8")
        self.assertIn("<dev_ctrl>", text)
        self.assertIn("@ dev_type         id     name       p_ctrl     p_real", text)
        self.assertIn("# wind_generator   1      wind       12.500000  11.500000", text)
        self.assertIn("# pv_generator     2      pv         7.500000   7.000000", text)
        self.assertIn("# estorage         3      storage    -4.000000  -3.500000", text)
        self.assertIn("# diesel_generator 4      diesel     80.000000  79.500000", text)
        self.assertIn("<yt_ctrl>", text)
        self.assertIn("@ id     name       rtu    pnt    value", text)
        self.assertIn("# 5      hydrogen   3      3      6.000000", text)
        self.assertIn("@ name                 value", text)
        self.assertIn("# agc_balance_mismatch 1.250000", text)

    def test_export_control_results_falls_back_to_p_ctrl_for_missing_real_power(self):
        output = Path("tmp_test/agc_unit_dev_ctrl_missing_real.e")
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            output.unlink()
        self.addCleanup(lambda: output.exists() and output.unlink())

        model = SimpleNamespace(
            wind_generator=[SimpleNamespace(id=1, name="wind", p_ctrl=3.5, run_stat=1)],
            pv_generator=[],
            estorage=[],
            diesel_generator=[SimpleNamespace(id=2, name="diesel", p_ctrl=0.0, run_stat=0)],
            yt=[],
            agc_balance_mismatch=0.0,
        )

        agc_ctrl.export_control_results(model, output)

        text = output.read_text(encoding="utf-8")
        self.assertIn("# wind_generator   1      wind       3.500000   3.500000", text)
        self.assertIn("# diesel_generator 2      diesel     0.000000   0.000000", text)

    def test_device_p_step_overrides_global_default(self):
        model = SimpleNamespace(para=[])
        device = SimpleNamespace(p_ctrl=0.0, p_cur=0.0, p_step=7.0)

        before, after = agc_ctrl._set_p_ctrl(model, device, 100.0, "renew", 0.0, 100.0)

        self.assertEqual(before, 0.0)
        self.assertEqual(after, 7.0)
        self.assertEqual(device.p_ctrl, 7.0)

    def test_storage_device_soc_dead_controls_charge_limit(self):
        model = SimpleNamespace(para=[])
        storage = SimpleNamespace(
            name="storage",
            run_stat=1,
            p_ctrl=0.0,
            p_cur=0.0,
            soc_cur=96.0,
            soc_min=10.0,
            soc_max=100.0,
            soc_dead=5.0,
            charge_p_max=30.0,
            dis_charge_p_max=30.0,
        )
        model.estorage = [storage]

        remaining = agc_ctrl._charge_storage(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 20.0)
        self.assertEqual(storage.p_ctrl, 0.0)

    def test_storage_charge_prefers_low_soc(self):
        low = SimpleNamespace(
            name="low",
            run_stat=1,
            p_ctrl=0.0,
            p_cur=0.0,
            soc_cur=20.0,
            soc_min=0.0,
            soc_max=100.0,
            charge_p_max=30.0,
            dis_charge_p_max=30.0,
        )
        high = SimpleNamespace(
            name="high",
            run_stat=1,
            p_ctrl=0.0,
            p_cur=0.0,
            soc_cur=80.0,
            soc_min=0.0,
            soc_max=100.0,
            charge_p_max=30.0,
            dis_charge_p_max=30.0,
        )
        model = SimpleNamespace(para=[], estorage=[high, low])

        remaining = agc_ctrl._charge_storage(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 0.0)
        self.assertEqual(low.p_ctrl, -20.0)
        self.assertEqual(high.p_ctrl, 0.0)

    def test_storage_discharge_prefers_high_soc(self):
        low = SimpleNamespace(
            name="low",
            run_stat=1,
            p_ctrl=0.0,
            p_cur=0.0,
            soc_cur=20.0,
            soc_min=0.0,
            soc_max=100.0,
            charge_p_max=30.0,
            dis_charge_p_max=30.0,
        )
        high = SimpleNamespace(
            name="high",
            run_stat=1,
            p_ctrl=0.0,
            p_cur=0.0,
            soc_cur=80.0,
            soc_min=0.0,
            soc_max=100.0,
            charge_p_max=30.0,
            dis_charge_p_max=30.0,
        )
        model = SimpleNamespace(para=[], estorage=[low, high])

        remaining = agc_ctrl._discharge_storage(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 0.0)
        self.assertEqual(high.p_ctrl, 20.0)
        self.assertEqual(low.p_ctrl, 0.0)

    def test_diesel_device_deadband_keeps_existing_storage_when_inside_band(self):
        model = SimpleNamespace(
            para=[],
            energyconsumer=[SimpleNamespace(p_fur=80.0, p_cur=80.0, run_stat=1)],
            wind_generator=[],
            pv_generator=[],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=5.0,
                    p_cur=5.0,
                    soc_cur=50.0,
                    soc_min=10.0,
                    soc_max=100.0,
                    charge_p_max=30.0,
                    dis_charge_p_max=30.0,
                )
            ],
            diesel_generator=[
                SimpleNamespace(
                    name="diesel",
                    run_stat=1,
                    p_ctrl=75.0,
                    p_cur=75.0,
                    p_min=80.0,
                    p_max=200.0,
                    p_dead=10.0,
                )
            ],
        )

        agc_ctrl._apply_diesel_min_output_coordination(logging.getLogger("test"), model)

        self.assertEqual(model.diesel_generator[0].p_ctrl, 75.0)
        self.assertEqual(model.estorage[0].p_ctrl, 5.0)

    def test_storage_above_max_minus_deadband_does_not_charge(self):
        model = SimpleNamespace(
            para=[SimpleNamespace(name="STORAGE_SOC_DEADBAND", value="5.0")],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    soc_cur=96.0,
                    soc_min=10.0,
                    soc_max=100.0,
                    charge_p_max=30.0,
                    dis_charge_p_max=30.0,
                )
            ],
        )

        remaining = agc_ctrl._charge_storage(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 20.0)
        self.assertEqual(model.estorage[0].p_ctrl, 0.0)

    def test_storage_below_min_plus_deadband_does_not_discharge(self):
        model = SimpleNamespace(
            para=[SimpleNamespace(name="STORAGE_SOC_DEADBAND", value="5.0")],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    soc_cur=14.0,
                    soc_min=10.0,
                    soc_max=100.0,
                    charge_p_max=30.0,
                    dis_charge_p_max=30.0,
                )
            ],
        )

        remaining = agc_ctrl._discharge_storage(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 20.0)
        self.assertEqual(model.estorage[0].p_ctrl, 0.0)

    def test_forced_storage_charge_respects_soc_deadband(self):
        model = SimpleNamespace(
            para=[SimpleNamespace(name="STORAGE_SOC_DEADBAND", value="5.0")],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    soc_cur=95.0,
                    soc_min=10.0,
                    soc_max=100.0,
                    charge_p_max=30.0,
                    dis_charge_p_max=30.0,
                )
            ],
        )

        remaining = agc_ctrl._force_storage_charge(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 20.0)
        self.assertEqual(model.estorage[0].p_ctrl, 0.0)

    def test_forced_storage_discharge_respects_soc_deadband(self):
        model = SimpleNamespace(
            para=[SimpleNamespace(name="STORAGE_SOC_DEADBAND", value="5.0")],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    soc_cur=15.0,
                    soc_min=10.0,
                    soc_max=100.0,
                    charge_p_max=30.0,
                    dis_charge_p_max=30.0,
                )
            ],
        )

        remaining = agc_ctrl._force_storage_discharge(logging.getLogger("test"), model, 20.0)

        self.assertEqual(remaining, 20.0)
        self.assertEqual(model.estorage[0].p_ctrl, 0.0)

    def test_diesel_deadband_uses_renewable_and_storage_without_moving_diesel(self):
        model = SimpleNamespace(
            para=[SimpleNamespace(name="DIESEL_MIN_DEADBAND", value="1.0")],
            energyconsumer=[SimpleNamespace(p_fur=180.0, p_cur=180.0, run_stat=1)],
            wind_generator=[
                SimpleNamespace(
                    name="wind",
                    run_stat=1,
                    p_ctrl=50.0,
                    p_cur=50.0,
                    p_fur=70.0,
                    p_min=0.0,
                    p_max=100.0,
                )
            ],
            pv_generator=[],
            estorage=[
                SimpleNamespace(
                    name="storage",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    soc_cur=50.0,
                    soc_min=0.0,
                    soc_max=100.0,
                    charge_p_max=40.0,
                    dis_charge_p_max=40.0,
                )
            ],
            diesel_generator=[
                SimpleNamespace(
                    name="diesel",
                    run_stat=1,
                    p_ctrl=80.0,
                    p_cur=80.0,
                    p_min=80.0,
                    p_max=200.0,
                )
            ],
        )

        agc_ctrl._apply_diesel_min_output_coordination(logging.getLogger("test"), model)

        self.assertEqual(model.diesel_generator[0].p_ctrl, 80.0)
        self.assertEqual(model.wind_generator[0].p_ctrl, 70.0)
        self.assertEqual(model.estorage[0].p_ctrl, 30.0)

    def test_renewable_dispatch_is_proportional_to_available_output(self):
        model = SimpleNamespace(
            para=[],
            wind_generator=[
                SimpleNamespace(
                    name="wind",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    p_fur=100.0,
                    p_min=0.0,
                    p_max=100.0,
                )
            ],
            pv_generator=[
                SimpleNamespace(
                    name="pv",
                    run_stat=1,
                    p_ctrl=0.0,
                    p_cur=0.0,
                    p_fur=300.0,
                    p_min=0.0,
                    p_max=300.0,
                )
            ],
        )

        remaining = agc_ctrl._dispatch_renewable_priority(logging.getLogger("test"), model, 200.0)

        self.assertEqual(remaining, 0.0)
        self.assertEqual(model.wind_generator[0].p_ctrl, 50.0)
        self.assertEqual(model.pv_generator[0].p_ctrl, 150.0)


if __name__ == "__main__":
    unittest.main()
