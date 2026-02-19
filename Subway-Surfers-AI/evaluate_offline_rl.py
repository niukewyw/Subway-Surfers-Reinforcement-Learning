import numpy as np
import gymnasium as gym
from d3rlpy.algos import CQL
from subway_surfers_env import SubwaySurfersEnv

def evaluate(model_path, n_episodes=5):
    # 加载模型
    cql = CQL()
    cql.build_with_env(SubwaySurfersEnv())  # 需要用环境构建网络结构
    cql.load_model(model_path)

    env = SubwaySurfersEnv(render_mode="human")  # 开启渲染以便观察
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            # 预处理观察：扁平化并归一化
            obs_flat = obs.flatten().astype(np.float32) / 255.0
            obs_flat = obs_flat.reshape(1, -1)  # 添加batch维度
            action = cql.predict(obs_flat)[0]    # 返回离散动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            env.render()
        print(f"Episode {episode+1}: 步数 {step}, 总奖励 {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    evaluate('cql_subway_final.pt', n_episodes=5)