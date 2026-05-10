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
        )

        updated = with_legacy_label(source, "new", side="hybrid")

        self.assertEqual("hybrid", updated.side)
        self.assertEqual("voltage", updated.kind)
        self.assertEqual("ACNode", updated.device_type)
        self.assertEqual("n1", updated.device_name)
        self.assertEqual("from", updated.terminal)
        self.assertEqual("magnitude", updated.component)
        self.assertEqual("new", updated.legacy_label)
        self.assertEqual("ac", source.side)
        self.assertEqual("old", source.legacy_label)


if __name__ == "__main__":
    unittest.main()
