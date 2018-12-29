# coding:utf-8
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
In part, Modified from OpenAI gym
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class CartPoleBalanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.GRAVITY = 9.8
        self.DT = 0.02
        # Physical params
        self.MASS_CART = 1.0
        self.MASS_POLE = 0.1
        self.MASS_TOTAL = (self.MASS_POLE + self.MASS_CART)
        self.WIDTH_CART = 0.25  # half of width
        self.LEN_POLE = 1.0
        self.COM_POLE = 0.5 * self.LEN_POLE
        self.MASSLEN_POLE = (self.MASS_POLE * self.COM_POLE)
        # Limitation
        self.MAX_X = 2.5
        self.MAX_VEL_X = 5.0 * np.sqrt(2.0)
        self.MAX_VEL_ANG = 2.5 * np.pi
        self.MAX_FORCE = 10.0

        # Create spaces
        high = np.array([self.MAX_X, 1.0, 1.0, self.MAX_VEL_X, self.MAX_VEL_ANG])
        self.action_space = spaces.Box(low=-self.MAX_FORCE, high=self.MAX_FORCE, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Initialize
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        # self.state[1] += np.pi  # difference is here
        return self._get_obs()

    def step(self, action):
        s = self.state
        force = np.clip(action[0], -self.MAX_FORCE, self.MAX_FORCE)

        ns = self._dynamics(s, force, self.DT)

        ns[2] = np.clip(ns[2], -self.MAX_VEL_X, self.MAX_VEL_X)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_ANG, self.MAX_VEL_ANG)
        ns[1] = self._angle_normalize(ns[1])

        collision = [False, False]
        collision[0] = np.absolute(ns[0]) > self.MAX_X
        if collision[0]:
            ns[0] = np.copysign(self.MAX_X, ns[0])
            ns[2] = 0.0
        collision[1] = np.absolute( np.sin(ns[1]) * self.LEN_POLE + ns[0] ) > self.MAX_X + self.WIDTH_CART
        if collision[1]:
            nth = np.arcsin( (np.copysign(self.MAX_X + self.WIDTH_CART, ns[0]) - ns[0]) / self.LEN_POLE )
            ns[1] = nth if np.cos(ns[1]) > 0.0 else np.sign(nth) * (np.pi - np.absolute(nth))
            ns[3] *= - 1.0

        self.state = ns

        # reward & punishment design
        reward = 0.0
        punish = 0.0
        done = False
        if any(collision):
            punish += 1.0
        reward += np.exp( - 1.0 * np.absolute(ns[1]) )

        return self._get_obs(), reward, done, {"punish": punish, "collision": 1 if any(collision) else 0}

    def _get_obs(self):
        s = self.state
        return np.array([s[0], np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def _dynamics(self, s, a, dt):
        x, theta, dx, dtheta = s

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (a + self.MASSLEN_POLE * dtheta * dtheta * sintheta) / self.MASS_TOTAL
        ddtheta = (self.GRAVITY * sintheta - costheta * temp) / (self.COM_POLE * (4.0/3.0 - self.MASS_POLE * costheta * costheta / self.MASS_TOTAL))
        ddx  = temp - self.MASSLEN_POLE * ddtheta * costheta / self.MASS_TOTAL

        return s + np.array([dx, dtheta, ddx, ddtheta]) * dt

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = 2.0 * (self.MAX_X + self.WIDTH_CART)
        scale = screen_width/world_width
        carty = 250 # TOP OF CART
        polelen = scale * self.LEN_POLE
        polewidth = polelen * 0.05
        cartwidth = scale * self.WIDTH_CART
        cartheight = scale * self.WIDTH_CART * 0.5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth, cartwidth, cartheight, -cartheight
            axleoffset =cartheight * 0.5
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth, polewidth, polelen - polewidth, -polewidth
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(self.track)

        s = self.state
        cartx = s[0]*scale + 0.5*screen_width # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-s[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
