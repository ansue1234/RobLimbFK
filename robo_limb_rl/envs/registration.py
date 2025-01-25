from gymnasium.envs.registration import register

register(
    id="LimbEnv-v0",
    entry_point="robo_limb_rl.envs.limb_env:LimbEnv",
    max_episode_steps=500,
    reward_threshold=10000,
)

register(
    id="SafeLimbEnv-v0",
    entry_point="robo_limb_rl.envs.limb_env:SafeLimbEnv",
    max_episode_steps=500,
    reward_threshold=10000,
)