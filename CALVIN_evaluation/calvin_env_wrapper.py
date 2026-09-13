"""Raw relative-action wrapper adapted from better_openpi/examples/calvin.

CALVIN environment and EGL helpers are provided by the external CALVIN checkout.
"""
import logging
import os

import gym
import numpy as np
import torch
from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id


class CalvinEnvWrapperRaw(gym.Wrapper):
    def __init__(self, validation_path):
        if "EGL_VISIBLE_DEVICES" not in os.environ:
            try:
                egl_id = get_egl_device_id(torch.cuda.current_device())
            except EglDeviceNotFoundError:
                logging.warning("EGL device mapping unavailable; using EGL_VISIBLE_DEVICES=0")
                egl_id = 0
            os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        observation_space = {
            "rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [],
            "state_obs": ["robot_obs"], "actions": ["rel_actions"],
            "language": ["language"],
        }
        super().__init__(get_env(validation_path, show_gui=False, obs_space=observation_space))

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("CALVIN relative action must contain seven finite values")
        return self.env.step(action)

    def reset(self, robot_obs=None, scene_obs=None):
        return self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    def get_obs(self):
        return self.env.get_obs()

    def get_info(self):
        return self.env.get_info()
