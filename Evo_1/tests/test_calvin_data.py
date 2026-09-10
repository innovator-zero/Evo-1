import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from dataset.dataset_process_suite import get_suite
from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset


CONFIG = Path(__file__).resolve().parents[1] / "dataset/config_calvin.yaml"


class CalvinDataTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG.read_text())
        self.entry = self.config["data_groups"]["calvin_franka_delta"]["InternData-Calvin_ABC"]
        self.suite = get_suite("custom", self.entry["suite_config"])
        self.rows = pd.DataFrame({
            "state.ee_pos": [[1., 2., 3.], [4., 5., 6.]],
            "state.ee_rot": [[.1, .2, .3], [.4, .5, .6]], "state.gripper": [[-1.], [1.]],
            "action.delta_ee_pos": [[-.1, .2, .3], [.4, -.5, .6]],
            "action.delta_ee_rot": [[.7, .8, -.9], [-.3, .2, .1]], "action.gripper": [[-1.], [1.]],
            "timestamp": [0., .1], "task_index": [0, 0],
        })

    def test_exact_openpi_channel_order_and_no_delta_conversion(self):
        result = self.suite.process(self.rows, use_delta_action=False)
        np.testing.assert_allclose(result.state, [1, 2, 3, .1, .2, .3, -1])
        expected = torch.cat([torch.tensor(np.stack(self.rows[k])) for k in
                              ("action.delta_ee_pos", "action.delta_ee_rot", "action.gripper")], dim=1)
        np.testing.assert_allclose(result.actions, expected.numpy())
        self.assertEqual((result.state_dim, result.action_dim), (7, 7))

    def stats(self):
        return {key: {metric: [float(i)] * len(self.rows[key][0]) for metric in
                      ("min", "max", "mean", "std", "q01", "q99")}
                for i, key in enumerate(self.entry["suite_config"]["state_concat_keys"] +
                                        self.entry["suite_config"]["action_concat_keys"])}

    def test_stats_order_and_source_unchanged(self):
        source = self.stats()
        original = copy.deepcopy(source)
        mapped = self.suite.adapt_stats(source)
        self.assertEqual(mapped["observation.state"]["mean"], [0, 0, 0, 1, 1, 1, 2])
        self.assertEqual(mapped["action"]["mean"], [3, 3, 3, 4, 4, 4, 5])
        self.assertEqual(source, original)
        del source["action.gripper"]
        with self.assertRaisesRegex(ValueError, "Missing raw"):
            self.suite.adapt_stats(source)

    def test_single_field_suite_unchanged(self):
        rows = pd.DataFrame({"observation.state": [[1., 2.]], "action": [[3., 4.]]})
        result = get_suite("custom").process(rows, use_delta_action=False)
        np.testing.assert_equal(result.state, [1., 2.])
        np.testing.assert_equal(result.actions, [[3., 4.]])

    def test_window_count_padding_masks_and_cache_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data_source"
            (root / "meta").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test command"}\n')
            stats = self.stats()
            for key, values in stats.items():
                dim = len(values["min"])
                values["min"], values["max"] = [-10.] * dim, [10.] * dim
            original_stats = json.dumps(stats)
            (root / "meta/stats.json").write_text(original_stats)
            for idx in range(2):
                self.rows.to_parquet(root / f"data/chunk-000/episode_{idx:06d}.parquet")
            self.entry["path"] = str(root)
            self.config["strict_data"] = False  # No video files needed for record-level assertions.
            kwargs = dict(config=self.config, cache_dir=Path(tmp) / "cache", action_horizon=3)
            dataset = LeRobotDataset(**kwargs, max_episodes=1)
            self.assertEqual(len(dataset), 2)
            record = dataset._read_record(1)
            expected = np.asarray(self.suite.process(self.rows.iloc[1:], False).actions)[0]
            np.testing.assert_allclose(record["action"], np.tile(expected, (3, 1)))
            state, smask, action, amask = dataset._prepare_state_action(record)
            self.assertEqual(tuple(state.shape), (24,))
            self.assertEqual(tuple(action.shape), (3, 24))
            self.assertEqual(int(smask.sum()), 7)
            self.assertEqual(int(amask.sum()), 21)
            self.assertTrue(torch.all(action[:, 7:] == 0))
            item = dataset[0]
            self.assertEqual(item["state"].dtype, torch.bfloat16)
            self.assertEqual(item["action"].dtype, torch.bfloat16)
            full = LeRobotDataset(**kwargs)
            self.assertEqual(len(full), 4)
            self.assertNotEqual(full.export_key, dataset.export_key)
            kwargs["action_horizon"] = 1
            shorter = LeRobotDataset(**kwargs, max_episodes=1)
            self.assertEqual(len(shorter), 2)
            self.assertNotEqual(shorter.export_key, dataset.export_key)
            self.assertEqual((root / "meta/stats.json").read_text(), original_stats)
            for ds in (dataset, full, shorter):
                ds._close_runtime_resources()

    def test_missing_field(self):
        with self.assertRaises(KeyError):
            self.suite.process(self.rows.drop(columns=["state.ee_rot"]), False)


if __name__ == "__main__":
    unittest.main()
