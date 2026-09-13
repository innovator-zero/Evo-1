import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from CALVIN_evaluation.evaluation import evaluate_policy, rollout, summarize
from CALVIN_evaluation.protocol import observation_request, validate_actions, validate_request


def observation():
    return {"robot_obs": np.arange(15, dtype=float), "rgb_obs": {
        "rgb_static": np.full((4, 5, 3), [255, 20, 0], dtype=np.uint8),
        "rgb_gripper": np.full((3, 3, 3), [0, 10, 255], dtype=np.uint8)}}


class FakeEnv:
    def __init__(self):
        self.actions, self.resets = [], 0

    def reset(self, **kwargs):
        self.resets += 1

    def get_obs(self):
        return observation()

    def get_info(self):
        return {"step": len(self.actions)}

    def step(self, action):
        self.actions.append(action.copy())
        return self.get_obs(), 0, False, self.get_info()


class FakeModel:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def reset(self):
        pass

    def step(self, obs, instruction):
        self.calls.append(instruction)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("connection lost")
        chunk = np.zeros((3, 7))
        chunk[:, 0] = [1, 2, 3]
        chunk[:, -1] = [-0.2, 0, 0.3]
        return chunk


class Oracle:
    def __init__(self, after=1, failed_task=None):
        self.after, self.failed_task = after, failed_task

    def get_task_info_for_set(self, start, current, tasks):
        if self.failed_task in tasks:
            return set()
        return tasks if current["step"] - start["step"] >= self.after else set()


class EvaluationTests(unittest.TestCase):
    def test_observation_contract(self):
        data = observation_request(observation(), "open drawer")
        self.assertEqual(data["state"], [0, 1, 2, 3, 4, 5, 14])
        self.assertEqual(data["image"][0][0], [255, 20, 0])
        self.assertEqual(data["wrist_image"][0][0], [0, 10, 255])
        data["state"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_invalid_actions_and_images(self):
        for bad in ([], [[0] * 24], [[float("nan")] * 7]):
            with self.assertRaises(ValueError):
                validate_actions(bad)
        for bad in ([], [[1]], np.full((2, 2, 3), 256), np.full((2, 2, 3), .5)):
            data = observation_request(observation(), "test")
            data["image"] = bad
            with self.assertRaises(ValueError):
                validate_request(data)

    def test_full_and_partial_chunks_and_gripper(self):
        for execute, expected in ((None, [1, 2, 3, 1]), (2, [1, 2, 1, 2])):
            env, model = FakeEnv(), FakeModel()
            self.assertTrue(rollout(env, model, Oracle(after=4), "a", "a",
                                    execute_steps=execute))
            self.assertEqual([a[0] for a in env.actions], expected)
            self.assertEqual(env.actions[0][-1], -1)
            self.assertEqual(env.actions[1][-1], 1)
            self.assertEqual(len(model.calls), 2)

    def test_720_step_limit(self):
        env, model = FakeEnv(), FakeModel()
        self.assertFalse(rollout(env, model, Oracle(after=721), "a", "a"))
        self.assertEqual(len(env.actions), 720)
        self.assertEqual(len(model.calls), 240)

    def test_chain_reset_queue_and_stop(self):
        env, model = FakeEnv(), FakeModel()
        tasks = ["a", "b", "c", "d", "e"]
        with tempfile.TemporaryDirectory() as directory:
            data = evaluate_policy(env, model, Oracle(failed_task="c"), [(0, tasks)],
                                   {t: [t] for t in tasks}, lambda _: (0, 0), directory,
                                   max_steps=2)
            self.assertEqual(env.resets, 1)
            self.assertEqual(model.calls, ["a", "b", "c"])
            self.assertEqual([a[0] for a in env.actions], [1, 1, 1, 2])
            self.assertEqual(data["avg_seq_len"], 2)
            self.assertEqual(data["task_info"]["c"], {"success": 0, "total": 1})
            self.assertNotIn("d", data["task_info"])
            self.assertTrue((Path(directory) / "success_rate.txt").is_file())
            with self.assertRaises(FileExistsError):
                evaluate_policy(env, model, Oracle(), [(0, tasks)], {}, lambda _: (0, 0), directory)

    def test_failure_preserves_completed_sequences(self):
        tasks = ["a"] * 5
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                evaluate_policy(FakeEnv(), FakeModel(fail_at=6), Oracle(),
                                [(0, tasks), (1, tasks)], {"a": ["a"]}, lambda _: (0, 0), directory)
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["num_completed"], 1)
            self.assertEqual(result["avg_seq_len"], 5)
            self.assertEqual(len((Path(directory) / "sequences.jsonl").read_text().splitlines()), 1)

    def test_chain_rates(self):
        result = summarize([0, 2, 5], [(0, list("abcde"))] * 3)
        self.assertAlmostEqual(result["avg_seq_len"], 7 / 3)
        self.assertAlmostEqual(result["chain_sr"]["2"], 2 / 3)
        self.assertAlmostEqual(result["chain_sr"]["3"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
