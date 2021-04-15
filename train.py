from game import FPS, GAME_PERIOD_SEC
from self_driving import SelfDriving
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback


env = SelfDriving()
model = SAC(
    'MlpPolicy',
    env,
    use_sde=True,
    verbose=1,
    policy_kwargs={'net_arch': [400, 300]},
    seed=1234
)

# model = SAC(
#     'MlpPolicy',
#     env,
#     learning_rate=0.0001,
#     learning_starts=GAME_PERIOD_SEC * FPS * 10,
#     gamma=0.999,
#     use_sde=True,
#     use_sde_at_warmup=True,
#     verbose=1,
#     policy_kwargs={'net_arch': [256, 256, 256]}
#     seed=1234
# )

model.learn(total_timesteps=10_000_000, log_interval=10, callback=CheckpointCallback(save_freq=GAME_PERIOD_SEC * FPS * 10, save_path='log', name_prefix='self-driving'))
model.save('self-driving')
