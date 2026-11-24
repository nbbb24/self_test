import copy
import json
import tempfile
import unittest
from pathlib import Path

from vllm_layout import load_layout


class VllmLayoutTests(unittest.TestCase):
    def test_glm_keeps_quantized_indexer_scales(self):
        layout = load_layout(Path(__file__).parent / "layouts/glm-5.2.json", 32)
        self.assertTrue(layout["metadata"]["prepare_value_verified"])
        self.assertEqual([(g["keys"], len(g["segments"])) for g in layout["groups"]], [(32, 198)])
        sizes = [s["bytes"] for s in layout["groups"][0]["segments"]]
        self.assertEqual(sizes.count(128 * 4), 21)
        self.assertEqual(sum(sizes), 11856384)

    def test_flash_keeps_group_calls_and_reachable_swa(self):
        layout = load_layout(Path(__file__).parent / "layouts/deepseek-v4-flash.json", 32)
        self.assertEqual(layout["metadata"]["request_tokens"], 16384)
        self.assertEqual(
            [(g["keys"], len(g["segments"])) for g in layout["groups"]], [(32, 84), (1, 20), (1, 22), (1, 21)]
        )
        self.assertEqual(layout["metadata"]["group_blocks"]["0"]["tokens_per_block"], 512)
        self.assertEqual(layout["metadata"]["group_blocks"]["1"]["tokens_per_block"], 16384)
        self.assertEqual(layout["metadata"]["selected_chunks"]["2"], [127])
        self.assertEqual(layout["metadata"]["selected_chunks"]["3"], [127])
        self.assertEqual(sum(g["keys"] * sum(s["bytes"] for s in g["segments"]) for g in layout["groups"]), 119046144)

    def test_rejects_replay_outside_registered_region(self):
        original = load_layout(Path(__file__).parent / "layouts/glm-5.2.json", 32)
        layout = copy.deepcopy(original)
        segment = layout["groups"][0]["segments"][0]
        segment["offset"] = layout["regions"][segment["region"]]["bytes"]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "invalid.json"
            target.write_text(json.dumps(layout))
            with self.assertRaisesRegex(ValueError, "segment exceeds region"):
                load_layout(target, 32)


if __name__ == "__main__":
    unittest.main()
