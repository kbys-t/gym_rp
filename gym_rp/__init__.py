# import logging
from gym.envs.registration import registry, register, make, spec

# logger = logging.getLogger(__name__)

register(
    id='CartPoleRP-v0',
    entry_point='gym_rp.envs:CartPoleBalanceEnv',
)

register(
    id='CartPoleRP-v1',
    entry_point='gym_rp.envs:CartPoleSwingEnv',
)

register(
    id='AcrobotRP-v0',
    entry_point='gym_rp.envs:AcrobotStrongEnv',
)

register(
    id='AcrobotRP-v1',
    entry_point='gym_rp.envs:AcrobotWeakEnv',
)

register(
    id='PlatoonRP-v0',
    entry_point='gym_rp.envs:PlatoonVelocityEnv',
)

register(
    id='PlatoonRP-v1',
    entry_point='gym_rp.envs:PlatoonAccelEnv',
)

register(
    id='SwimmerRP-v0',
    entry_point='gym_rp.envs:SwimmerStraightEnv',
)

register(
    id='SwimmerRP-v1',
    entry_point='gym_rp.envs:SwimmerTurnEnv',
)
