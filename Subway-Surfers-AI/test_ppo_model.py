from stable_baselines3 import PPO
from subway_surfers_env import SubwaySurfersEnv

env = SubwaySurfersEnv(render_mode="rgb_array")  # 开启渲染观察
model = PPO.load('./ppo_models/subway_ppo_100000.zip')

obs, info = env.reset()
total_reward = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode finished. Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0
    env.render()