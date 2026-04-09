from gymnasium.envs.registration import register

from .event_env import EventEnv  # noqa: F401

register(
    id="EventEnv-v0",
    entry_point="rl_captcha.environment.event_env:EventEnv",
)
