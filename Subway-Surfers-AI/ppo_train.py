from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from subway_surfers_env import SubwaySurfersEnv
import torch
# 创建环境（注意：SB3需要向量环境，但单环境也可以）
env = SubwaySurfersEnv(render_mode=None)  # 无头模式，加快速度
env = DummyVecEnv([lambda: env])  # 包装为向量环境（可选，但推荐）

# 设置模型保存回调：每100000步保存一次
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='./ppo_models/',
    name_prefix='subway_ppo'
)

# 创建PPO模型，使用CNN策略处理图像输入
model = PPO(
    'CnnPolicy',
    env,
    verbose=1,
    tensorboard_log='./ppo_tensorboard/',
    learning_rate=3e-4,
    n_steps=2048,          # 每次更新前收集的步数
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 开始训练
model.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback,
    progress_bar=True      # 显示进度条
)

# 保存最终模型
model.save('ppo_subway_final')