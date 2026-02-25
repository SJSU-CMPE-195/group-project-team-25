from gymnasium.envs.registration import register

from .event_env import EventEnv

register(
    id="EventEnv-v0",
    entry_point="rl_captcha.environment.event_env:EventEnv",
)