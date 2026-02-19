import pickle
import time
import numpy as np
from subway_surfers_env import SubwaySurfersEnv

def collect_random(total_steps=80000, save_interval=10000):
    """
    收集随机策略数据，占总数据的80%（例如8万步）
    """
    env = SubwaySurfersEnv(render_mode=None)
    data = []
    step = 0
    episode_reward = 0
    obs, info = env.reset()

    while step < total_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data.append({
            "obs": obs,
            "action": int(action),
            "reward": float(reward),
            "next_obs": next_obs,
            "done": done
        })

        episode_reward += reward
        obs = next_obs
        step += 1

        if step % save_interval == 0:
            with open(f"random_data_{step}.pkl", "wb") as f:
                pickle.dump(data, f)
            print(f"随机数据已保存 {step} 步，当前片段奖励 {episode_reward:.2f}")

        if done:
            print(f"随机片段结束，总奖励 {episode_reward:.2f}")
            obs, info = env.reset()
            episode_reward = 0

    env.close()
    with open("random_data_final.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"随机数据收集完成，总步数 {len(data)}")

if __name__ == "__main__":
    collect_random(total_steps=80000)  # 收集8万步随机数据