# coding:utf-8
"""
Multi-link swimmer moving in a viscous fluid, Coulom 2002.
In part, Copied from OpenAI gym and RLpy
https://github.com/rlpy/rlpy
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class SwimmerTurnEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.DT = 0.05
        # Physical params
        self.N_LINK = 4  # number of links
        self.N_JOINT = self.N_LINK - 1
        self.K1 = 15.0   #
        self.K2 = 0.05   #
        self.MASSES = np.ones(self.N_LINK) # masses of all links
        self.LENGTHS = np.ones(self.N_LINK) # lengths of all links
        self.INERTIA = self.MASSES * self.LENGTHS**2 / 12.0

        Q_ = np.eye(self.N_LINK, k=1) - np.eye(self.N_LINK)
        Q_[-1, :] = self.MASSES
        A_ = np.eye(self.N_LINK, k=1) + np.eye(self.N_LINK)
        A_[-1, -1] = 0.0
        self.P_ = 0.5 * np.dot(np.linalg.inv(Q_), A_ * self.LENGTHS[None, :])
        self.U_ = np.eye(self.N_LINK) - np.eye(self.N_LINK, k=-1)
        self.U_ = self.U_[:, :-1]
        self.G_ = np.dot(self.P_.T * self.MASSES[None, :], self.P_)

        # Limitation
        self.MAX_POS = 5.0
        self.MAX_ANG = 1.0 * np.pi
        self.MAX_ANG_VEL = 2.0 * np.pi
        self.MAX_TORQUE = 2.0 * np.pi

        # Create spaces
        high_a = np.array([self.MAX_TORQUE] * self.N_JOINT)
        high_s = np.array([self.MAX_POS] * 2 + [self.MAX_ANG] * self.N_JOINT + [self.MAX_ANG_VEL] * self.N_JOINT)
        self.action_space = spaces.Box(-high_a, high_a, dtype=np.float32)
        self.observation_space = spaces.Box(-high_s, high_s, dtype=np.float32)

        # Initialize
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # no disturbance in initial states for simplicity
        self.theta = np.zeros((self.N_LINK,))
        self.dtheta = np.zeros((self.N_LINK,))
        self.pos_cm = np.zeros(2)
        self.vel_cm = np.zeros(2)
        self.target = np.array([self.MAX_POS, 0.0]) * 0.75   # difference is here
        return self._get_obs()

    def step(self, action):
        s = np.hstack((self.pos_cm, self.theta, self.vel_cm, self.dtheta))
        a = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)

        ns = self._dynamics(s, a, self.DT)

        self.pos_cm = ns[:2]
        self.theta = ns[2 : 2 + self.N_LINK]
        q_ = self.theta[1:] - self.theta[:-1]
        collision = np.absolute(q_) > self.MAX_ANG
        q_ = np.clip(q_, -self.MAX_ANG, self.MAX_ANG)
        self.theta[1:] = self.theta[:-1] + q_
        self.vel_cm = ns[2 + self.N_LINK : 4 + self.N_LINK]
        self.dtheta = ns[4 + self.N_LINK : 4 + 2 * self.N_LINK]
        self.dtheta[1:] = np.clip(self.dtheta[1:], -self.MAX_ANG_VEL, self.MAX_ANG_VEL)
        work = np.maximum(0.0, a * self.dtheta[1:] * self.DT).sum()

        obs = self._get_obs()
        # reward & punishment design
        reward = 0.0
        punish = 0.0
        done = False
        done = np.absolute(self.pos_cm[0]) > self.MAX_POS or np.absolute(self.pos_cm[1]) > self.MAX_POS
        punish += 1.0 if any(collision) else 0.0
        reward += np.exp(- 0.5 * np.linalg.norm(obs[:2]))

        return obs, reward, done, {"punish": punish, "collision": 1 if any(collision) else 0, "work": work}

    def _get_obs(self):
        # for relative position
        T = np.empty((self.N_LINK, 2))
        T[:, 0] = np.cos(self.theta)
        T[:, 1] = np.sin(self.theta)
        R = np.dot(self.P_, T)
        R1 = R - 0.5 * self.LENGTHS[:, None] * T
        R2 = R + 0.5 * self.LENGTHS[:, None] * T
        self.lines = np.array([[R1[i], R2[i]] for i in range(self.N_LINK)]).reshape((-1, 2)) + self.pos_cm.reshape((-1, 2))
        dp = self.target - self.lines[0]
        cth = np.cos(self.theta[0])
        sth = np.sin(self.theta[0])
        c2n_x = np.array([cth, sth])
        c2n_y = np.array([-sth, cth])
        dp = np.array([np.sum(dp * c2n_x), np.sum(dp * c2n_y)])
        # for angles
        q_ = self.theta[1:] - self.theta[:-1]
        return np.hstack((dp, q_, self.dtheta[1:]))

    def _dynamics(self, s, a, dt):
        d = self.N_LINK
        P = self.P_
        I = self.INERTIA
        G = self.G_
        U = self.U_
        lengths = self.LENGTHS
        masses = self.MASSES
        k1 = self.K1
        k2 = self.K2

        theta = s[2 : 2 + d]
        vcm = s[2 + d : 4 + d]
        dtheta = s[4 + d : 4 + 2 * d]

        cth = np.cos(theta)
        sth = np.sin(theta)
        rVx = np.dot(P, -sth * dtheta)
        rVy = np.dot(P, cth * dtheta)
        Vx = rVx + vcm[0]
        Vy = rVy + vcm[1]

        Vn = -sth * Vx + cth * Vy
        Vt = cth * Vx + sth * Vy

        EL1 = np.dot((self._v1Mv2(-sth, G, cth) + self._v1Mv2(cth, G, sth)) * dtheta[None, :]
            + (self._v1Mv2(cth, G, -sth) + self._v1Mv2(sth, G, cth)) * dtheta[:, None], dtheta)
        EL3 = np.diag(I) + self._v1Mv2(sth, G, sth) + self._v1Mv2(cth, G, cth)
        EL2 = - k1 * np.dot((self._v1Mv2(-sth, P.T, -sth) + self._v1Mv2(cth, P.T, cth)) * lengths[None, :], Vn) \
            - k1 * np.power(lengths, 3) * dtheta / 12.0 \
            - k2 * np.dot((self._v1Mv2(-sth, P.T, cth) + self._v1Mv2(cth, P.T, sth)) * lengths[None, :], Vt)

        ds = np.zeros_like(s)
        ds[:2] = vcm
        ds[2 : 2 + d] = dtheta
        ds[2 + d] = - (k1 * np.sum(-sth * Vn) + k2 * np.sum(cth * Vt)) / np.sum(masses)
        ds[3 + d] = - (k1 * np.sum(cth * Vn) + k2 * np.sum(sth * Vt)) / np.sum(masses)
        ds[4 + d : 4 + 2 * d] = np.linalg.solve(EL3, EL1 + EL2 + np.dot(U, a))

        return s + ds * dt

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def _v1Mv2(self, v1, M, v2):
        return v1[:, None] * M * v2[None, :]

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.MAX_POS,self.MAX_POS,-self.MAX_POS,self.MAX_POS)

        jtransform = rendering.Transform(rotation=0.0, translation=[self.target[0],self.target[1]])
        circ = self.viewer.draw_circle(0.2)
        circ.set_color(0.8, 0.8, 0)
        circ.add_attr(jtransform)

        self.viewer.draw_polyline(self.lines, linewidth=5)

        jtransform = rendering.Transform(rotation=0.0, translation=[self.lines[0,0],self.lines[0,1]])
        circ = self.viewer.draw_circle(0.05)
        circ.set_color(0.5, 0.5, 0.5)
        circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
