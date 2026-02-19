import pickle
import time
import numpy as np
from pynput import keyboard
from subway_surfers_env import SubwaySurfersEnv

class ManualRecorder:
    def __init__(self, total_steps=20000, step_delay=0.02):
        # 创建环境，使用human模式以便观察
        self.env = SubwaySurfersEnv(render_mode="rgb_array")
        # 尝试缩短滑动时间（如果游戏能识别0.1秒的滑动，可进一步减小延迟）
        self.env.swipe_duration = 0.10001   # 原为0.2，改为0.1加快响应
        # 环境内部的time.sleep(0.05)暂时无法去除，但0.05秒影响有限

        self.data = []
        self.step_count = 0
        self.total_steps = total_steps
        self.current_action = 0          # 当前按下的动作值
        self.need_action = False          # 是否需要执行一次动作（按键按下时设为True）
        self.running = True
        self.episode_reward = 0
        self.obs, self.info = self.env.reset()
        self.step_delay = step_delay

    def on_press(self, key):
        """按键按下时设置动作并标记需要执行"""
        try:
            if key == keyboard.Key.left:
                self.current_action = 1
                self.need_action = True
            elif key == keyboard.Key.right:
                self.current_action = 2
                self.need_action = True
            elif key == keyboard.Key.up:
                self.current_action = 3
                self.need_action = True
            elif key == keyboard.Key.down:
                self.current_action = 4
                self.need_action = True
        except AttributeError:
            pass

    def on_release(self, key):
        """释放键时无需重置动作，因为我们已经触发式执行"""
        if key == keyboard.Key.esc:
            self.running = False
            return False

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        print(f"触发式手动控制启动 | 步延迟 {self.step_delay}s | 滑动耗时 {self.env.swipe_duration}s")
        print("按下方向键触发一次滑动，ESC退出并保存数据")

        while self.running and self.step_count < self.total_steps:
            start_time = time.time()

            # 判断是否需要执行动作
            if self.need_action:
                action = self.current_action
                self.need_action = False   # 只触发一次
            else:
                action = 0

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 记录数据
            self.data.append({
                "obs": self.obs,
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs,
                "done": done
            })

            self.episode_reward += reward
            self.obs = next_obs
            self.step_count += 1

            if done:
                print(f"片段结束，奖励 {self.episode_reward:.2f}")
                self.obs, self.info = self.env.reset()
                self.episode_reward = 0

            # 精确控制循环频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.step_delay - elapsed)
            time.sleep(sleep_time)

        listener.stop()
        self.env.close()
        with open("manual_data.pkl", "wb") as f:
            pickle.dump(self.data, f)
        print(f"数据收集完成，共 {len(self.data)} 步")

if __name__ == "__main__":
    # 可以尝试更小的步延迟，例如0.01秒
    recorder = ManualRecorder(total_steps=20000, step_delay=0.0001)
    recorder.run()