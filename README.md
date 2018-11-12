# gym_rp

# Dependency

[OpenAI Gym](https://github.com/openai/gym)

# Installation

```bash
git clone https://github.com/kbys-t/gym_RP.git
cd gym_RP
pip install -e .
```

# How to use
1. First of all,
`import gym_rp`

1. Select environment from `["CartPoleRP-v0", "AcrobotRP-v0"]`...
```python
ENV_NAME = "AcrobotRP-v0"
env = gym.make(ENV_NAME)
```

1. Extract punishment info from `info`
```python
observation, reward, done, info = env.step(action)
punish = info["punish"]
```

1. Another info can also be extracted from `info`.
For example,
```python
observation, reward, done, info = env.step(action)
for key in info:
    criteria[epi][key] = criteria[epi].get(key, 0) + info[key]
```
