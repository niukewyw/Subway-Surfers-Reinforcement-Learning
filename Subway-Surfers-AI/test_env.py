from subway_surfers_env import SubwaySurfersEnv

env = SubwaySurfersEnv(render_mode="rgb_array")  # 开启可视化窗口
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, info)
    env.render()  # 刷新窗口
    if terminated:
        obs, info = env.reset()

env.close()  # 清理资源