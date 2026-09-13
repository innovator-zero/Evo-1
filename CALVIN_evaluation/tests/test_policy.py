import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from websockets.asyncio.server import serve

from CALVIN_evaluation.policy import CalvinPolicy, checkpoint_weights, validate_stats
from CALVIN_evaluation.protocol import ClientModel
from CALVIN_evaluation.server import handle_connection
from scripts.Evo1_server import Normalizer
from CALVIN_evaluation.tests.test_evaluation import observation


def stats():
    return {"calvin_franka_delta": {"InternData-Calvin_ABC": {
        key: {"min": [-2.] * 7, "max": [2.] * 7}
        for key in ("observation.state", "action")}}}


class PolicyTests(unittest.TestCase):
    def test_checkpoint_formats_and_missing_files(self):
        for name, key in (("mp_rank_00_model_states.pt", "module"),
                          ("checkpoint.pt", "model_state_dict")):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileNotFoundError):
                    checkpoint_weights(directory)
                torch.save({key: {"weight": torch.tensor([2.])}}, Path(directory) / name)
                self.assertEqual(checkpoint_weights(directory)["weight"].item(), 2)
                torch.save({}, Path(directory) / name)
                with self.assertRaisesRegex(ValueError, "missing model key"):
                    checkpoint_weights(directory)

    def test_stats_validation(self):
        values = stats()
        validate_stats(values, "calvin_franka_delta", "InternData-Calvin_ABC", "bounds")
        values["calvin_franka_delta"]["InternData-Calvin_ABC"]["action"]["min"] = [0]
        with self.assertRaisesRegex(ValueError, "seven finite"):
            validate_stats(values, "calvin_franka_delta", "InternData-Calvin_ABC", "bounds")

    def test_preprocessing_masks_and_denormalization(self):
        class Model:
            def run_inference(self, **inputs):
                self.inputs = inputs
                return torch.full((1, 2, 24), .5)
        model = Model()
        policy = CalvinPolicy(model, Normalizer(stats(), "bounds"),
                              SimpleNamespace(device="cpu", image_size=448,
                                              horizon=2, action_horizon=None))
        request = {"image": np.full((5, 4, 3), [255, 0, 0]),
                   "wrist_image": np.full((3, 3, 3), [0, 0, 255]),
                   "state": [1.] * 7, "prompt": "test"}
        actions = policy.infer(request)["actions"]
        np.testing.assert_allclose(actions, np.ones((2, 7)), atol=1e-6)
        inputs = model.inputs
        self.assertEqual(inputs["image_mask"].tolist(), [1, 1, 0])
        self.assertEqual(inputs["action_mask"].tolist(), [[1] * 7 + [0] * 17])
        self.assertEqual(inputs["images"][0].shape, (3, 448, 448))
        self.assertTrue(torch.all(inputs["images"][0][0] == 1))
        self.assertTrue(torch.all(inputs["images"][0][2] == 0))
        self.assertTrue(torch.all(inputs["images"][1][2] == 1))
        self.assertTrue(torch.all(inputs["images"][2] == 0))
        self.assertEqual(inputs["state_input"].tolist(), [[.5] * 7 + [0.] * 17])


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_socket_protocol_and_server_error(self):
        class Policy:
            def infer(self, data):
                if data["prompt"] == "fail":
                    raise ValueError("test failure")
                return {"actions": [[0.] * 7] * 3}
        async with serve(lambda ws: handle_connection(ws, Policy()), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]

            def request():
                client = ClientModel(port=port)
                try:
                    self.assertEqual(client.step(observation(), "ok").shape, (3, 7))
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        client.step(observation(), "fail")
                finally:
                    client.close()
            await asyncio.to_thread(request)


if __name__ == "__main__":
    unittest.main()
