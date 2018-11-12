# coding:utf-8
"""
Move to goal with formation
Goal moves along with sine curve
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class PlatoonVelocityEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.DT = 0.02
        # Physical params
        # Limitation
        self.MAX_VEL = 2.0
        self.MAX_ANG_VEL = 2.0 * np.pi
        self.MAX_POS = 1.0
        # reference
        self.AGENT_DISTANCE = 0.6
        self.TARGET_DISTANCE = 0.15
        self.TARGET_WAVE = np.array([10.0, 0.5])

        # Create spaces
        high_a = np.array([self.MAX_VEL, self.MAX_ANG_VEL, self.MAX_VEL, self.MAX_ANG_VEL])
        high_s = np.array([np.sqrt(2.0)*self.MAX_POS, 1.0, 1.0, np.sqrt(2.0)*self.MAX_POS, 1.0, 1.0])
        low_s = - high_s
        low_s[0] = 0.0
        low_s[3] = 0.0
        self.action_space = spaces.Box(-high_a, high_a, dtype=np.float32)
        self.observation_space = spaces.Box(low_s, high_s, dtype=np.float32)

        # Initialize
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(6, ))
        self.state[0] -= self.MAX_POS
        self.state[3] -= self.MAX_POS
        self.target = np.zeros((2, ))
        self.target[0] -= self.MAX_POS*2.0  # force to edge
        return self._get_obs()

    def step(self, action):
        s = self.state
        a = np.clip(action, self.action_space.low, self.action_space.high)
        # update the position of each agent
        ns = self._dynamics(s, a, self.DT)
        ns[0:2] = np.clip(ns[0:2], -self.MAX_POS, self.MAX_POS)
        ns[2] = self._angle_normalize(ns[2])
        ns[3:5] = np.clip(ns[3:5], -self.MAX_POS, self.MAX_POS)
        ns[5] = self._angle_normalize(ns[5])

        # convert to observatin
        self.state = ns
        obs = self._get_obs()

        # reward & punishment design
        reward = 0.0
        punish = 0.0
        done = False
        punish += 0.5 * (obs[3] > self.AGENT_DISTANCE) + 0.5 * (obs[4] < 0.0)
        failure = 1 if punish > 0.0 else 0
        reward += 1.0 if obs[0] < self.TARGET_DISTANCE and obs[1] > 0.0 else 0.0

        return obs, reward, done, {"punish": punish, "failure": failure}

    def _get_obs(self):
        s = self.state
        rtv = np.zeros((6, ))
        # distance and angle to target
        rth = self._angle_normalize( np.arctan2(self.target[1] - s[1], self.target[0] - s[0]) - s[2] )
        rtv[0] = np.linalg.norm(self.target - s[0:2])
        rtv[1] = np.cos(rth)
        rtv[2] = np.sin(rth)
        # distance and angle to forward agent
        rth = self._angle_normalize( np.arctan2(s[1] - s[4], s[0] - s[3]) - s[5] )
        rtv[3] = np.linalg.norm(s[0:2] - s[3:5])
        rtv[4] = np.cos(rth)
        rtv[5] = np.sin(rth)
        return rtv

    def _dynamics(self, s, a, dt):
        # http://myenigma.hatenablog.com/entry/20140301/1393648106
        th1 = s[2]
        v1 = a[0]
        w1 = a[1]
        th2 = s[5]
        v2 = a[2]
        w2 = a[3]

        dth1 = w1 * dt
        dx1 = v1 / w1 * (np.sin(th1 + dth1) - np.sin(th1)) if np.absolute(w1) > 1e-12 else v1 * np.cos(th1)
        dy1 = - v1 / w1 * (np.cos(th1 + dth1) - np.cos(th1)) if np.absolute(w1) > 1e-12 else v1 * np.sin(th1)
        dth2 = w2 * dt
        dx2 = v2 / w2 * (np.sin(th2 + dth2) - np.sin(th2)) if np.absolute(w2) > 1e-12 else v2 * np.cos(th2)
        dy2 = - v2 / w2 * (np.cos(th2 + dth2) - np.cos(th2)) if np.absolute(w2) > 1e-12 else v2 * np.sin(th2)

        # update the target position (difference is here)
        ysign = 1.0 if np.absolute(self.target[0] / self.MAX_POS) < 0.5 else -1.0
        self.target[1] = self.target[1] * np.cos(2.0*np.pi / self.TARGET_WAVE[0]*self.DT) + ysign * np.sqrt(self.TARGET_WAVE[1]**2 - self.target[1]**2) * np.sin(2.0*np.pi / self.TARGET_WAVE[0]*self.DT)
        self.target[0] += 2.0 * self.MAX_POS / self.TARGET_WAVE[0]*self.DT
        self.target = np.clip(self.target, -self.MAX_POS, self.MAX_POS)

        return s + np.array([dx1, dy1, dth1, dx2, dy2, dth2])

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)


    def render(self, mode='human'):
        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.MAX_POS,self.MAX_POS,-self.MAX_POS,self.MAX_POS)

        self.viewer.draw_line((-self.MAX_POS, 0), (self.MAX_POS, 0))
        self.viewer.draw_line((0, -self.MAX_POS), (0, self.MAX_POS))

        jtransform = rendering.Transform(rotation=0.0, translation=[self.target[0],self.target[1]])
        circ = self.viewer.draw_circle(0.02)
        circ.set_color(0.2, 0.2, 0.2)
        circ.add_attr(jtransform)

        r = 0.05
        age1 = self.viewer.draw_polygon([(-r,-r/1.5), (-r,r/1.5), (r,0), (r,0)])
        age1.set_color(0.4, 0.761, 0.647)
        jtransform = rendering.Transform(rotation=s[2], translation=[s[0], s[1]])
        age1.add_attr(jtransform)

        age2 = self.viewer.draw_polygon([(-r,-r/1.5), (-r,r/1.5), (r,0), (r,0)])
        age2.set_color(0.988, 0.553, 0.384)
        jtransform = rendering.Transform(rotation=s[5], translation=[s[3], s[4]])
        age2.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
