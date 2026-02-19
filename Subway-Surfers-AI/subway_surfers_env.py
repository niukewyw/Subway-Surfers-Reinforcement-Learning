import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import time
from mss import mss
import os
import subprocess
import easyocr

class SubwaySurfersEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, game_exe_path=r"D:\RL_personal_project\Subway-Surfers-AI\Subway_Surfers.exe",
                 stuck_threshold=10, enable_stuck_detection=True):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)

        self.render_mode = render_mode
        self.game_exe_path = game_exe_path
        self.game_process = None
        self.sct = mss()
        self.monitor = {"left": 937, "top": 485, "width": 685, "height": 623}

        self.center_x = self.monitor["left"] + self.monitor["width"] // 2
        self.center_y = self.monitor["top"] + self.monitor["height"] // 2

        self.swipe_offset = 100
        self.swipe_duration = 0.2

        self.ocr_reader = easyocr.Reader(['en'])

        self.mileage_region = {"left": 1659, "top": 443, "width": 1781-1659, "height": 480-443}
        self.coin_region = {"left": 1646, "top": 498, "width": 1754-1646, "height": 537-498}
        self.score_region = {"left": 1269, "top": 628, "width": 1503-1269, "height": 681-628}

        self.last_mileage = 0
        self.last_coin = 0

        # 按钮坐标（根据用户描述）
        self.play_button_center_death = (1401, 1123)  # 死亡画面play按钮
        self.play_button_center_main = (1418, 1074)   # 主菜单play按钮
        self.start_button_center = (1155, 1066)       # 开始游戏按钮

        # 防卡死机制
        self.enable_stuck_detection = enable_stuck_detection
        self.stuck_threshold = stuck_threshold
        self.stuck_counter = 0
        self.stuck_mode = False
        self.stuck_terminate_threshold = stuck_threshold * 3
        self.stuck_terminate_counter = 0
        self._last_termination_reason = "normal"   # 记录上次结束原因，用于reset

    def _get_obs(self):
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_LINEAR)
        return img

    def _ocr_digit(self, region):
        img = np.array(self.sct.grab(region))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        result = self.ocr_reader.readtext(binary, allowlist='0123456789')
        if result:
            text = result[0][1]
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                return int(digits)
        return 0

    def _get_info(self):
        mileage = self._ocr_digit(self.mileage_region)
        coin = self._ocr_digit(self.coin_region)
        score_val = self._ocr_digit(self.score_region)
        collision = (score_val > 0)
        info = {"mileage": mileage, "coin": coin, "collision": collision}
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置防卡死状态
        self.stuck_counter = 0
        self.stuck_mode = False
        self.stuck_terminate_counter = 0

        # 如果是因为卡死强制结束，则杀掉进程重新启动
        if self._last_termination_reason == "stuck":
            print("因卡死强制结束，杀死游戏进程重新启动")
            if self.game_process:
                self.game_process.kill()
                self.game_process = None
                time.sleep(2)  # 等待进程完全退出

        # 如果游戏进程不存在，则启动游戏
        if self.game_process is None or self.game_process.poll() is not None:
            self.game_process = subprocess.Popen(self.game_exe_path)
            print("等待游戏启动...")
            time.sleep(8)  # 等待加载

            # 自动点击主菜单play按钮和开始按钮
            pyautogui.click(*self.play_button_center_main)
            print("点击主菜单PLAY按钮")
            time.sleep(2)  # 等待过渡
            pyautogui.click(*self.start_button_center)
            print("点击开始按钮")
            time.sleep(3)  # 等待游戏真正开始

        else:
            # 游戏已在运行，通常处于死亡画面或主菜单，点击死亡画面的play按钮重启
            # 先激活窗口
            pyautogui.click(self.center_x, self.center_y)
            time.sleep(0.2)

            max_retries = 3
            for attempt in range(max_retries):
                pyautogui.click(*self.play_button_center_death)
                print(f"点击死亡PLAY重启 (尝试 {attempt+1}/{max_retries})")

                # 等待游戏开始
                start_time = time.time()
                game_started = False
                while time.time() - start_time < 10:
                    time.sleep(0.5)
                    info_check = self._get_info()
                    if not info_check["collision"] or info_check["mileage"] > 0:
                        game_started = True
                        print("游戏已成功开始")
                        break
                if game_started:
                    break
                else:
                    print(f"尝试 {attempt+1} 超时")
            else:
                print("警告：无法重新开始游戏")

        # 重置上一帧数值
        self.last_mileage = 0
        self.last_coin = 0

        # 获取初始观察
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # 防卡死检测
        if self.enable_stuck_detection:
            info_before = self._get_info()
            current_mileage = info_before["mileage"]

            if current_mileage == self.last_mileage:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            if not self.stuck_mode and self.stuck_counter >= self.stuck_threshold:
                self.stuck_mode = True
                self.stuck_terminate_counter = 0
                print("检测到卡死，进入无操作模式")
            elif self.stuck_mode and current_mileage != self.last_mileage:
                self.stuck_mode = False
                self.stuck_counter = 0
                self.stuck_terminate_counter = 0
                print("卡死解除")

            if self.stuck_mode:
                self.stuck_terminate_counter += 1
                action = 0

        # 执行动作
        if action != 0:
            pyautogui.moveTo(self.center_x, self.center_y)

        if action == 1:
            pyautogui.dragRel(-self.swipe_offset, 0, duration=self.swipe_duration)
        elif action == 2:
            pyautogui.dragRel(self.swipe_offset, 0, duration=self.swipe_duration)
        elif action == 3:
            pyautogui.dragRel(0, -self.swipe_offset, duration=self.swipe_duration)
        elif action == 4:
            pyautogui.dragRel(0, self.swipe_offset, duration=self.swipe_duration)

        time.sleep(0.005)

        observation = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)

        # 判断结束原因
        terminated = info["collision"]
        truncated = False

        # 卡死强制结束
        if self.enable_stuck_detection and self.stuck_mode and self.stuck_terminate_counter >= self.stuck_terminate_threshold:
            print("卡死时间过长，强制结束本回合")
            terminated = True
            self._last_termination_reason = "stuck"
        else:
            # 如果正常死亡，记录 reason 为 normal
            if terminated:
                self._last_termination_reason = "normal"

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, info):
        reward = 0.1
        mileage_gain = info["mileage"] - self.last_mileage
        if mileage_gain > 0:
            reward += mileage_gain * 0.01
        coin_gain = info["coin"] - self.last_coin
        if coin_gain > 0:
            reward += coin_gain * 0.1
        self.last_mileage = info["mileage"]
        self.last_coin = info["coin"]
        if info["collision"]:
            reward -= 10
        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        elif self.render_mode == "human":
            pass

    def close(self):
        if self.game_process:
            self.game_process.kill()
            self.game_process = None