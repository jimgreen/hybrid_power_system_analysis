import unittest


class StateMetadataTest(unittest.TestCase):
    def test_with_legacy_label_preserves_structured_fields(self):
        from secore.state_metadata import StateMeta, with_legacy_label

        source = StateMeta(
            side="ac",
            kind="voltage",
            device_type="ACNode",
            device_name="n1",
            terminal="from",
            component="magnitude",
            legacy_label="old",
            device_pos=3,
            device_type_code=1,
            meas_type_code=1,
        )

        updated = with_legacy_label(source, "new", side="hybrid")

        self.assertEqual("hybrid", updated.side)
        self.assertEqual("voltage", updated.kind)
        self.assertEqual("ACNode", updated.device_type)
        self.assertEqual("n1", updated.device_name)
        self.assertEqual("from", updated.terminal)
        self.assertEqual("magnitude", updated.component)
        self.assertEqual("new", updated.legacy_label)
        self.assertEqual(3, updated.device_pos)
        self.assertEqual(1, updated.device_type_code)
        self.assertEqual(1, updated.meas_type_code)
        self.assertEqual("ac", source.side)
        self.assertEqual("old", source.legacy_label)


if __name__ == "__main__":
    unittest.main()
