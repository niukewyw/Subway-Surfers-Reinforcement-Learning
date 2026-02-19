import pyautogui
import time

# 从你校准的环境参数中获取窗口中心
center_x = 937 + 685 // 2   # 根据你的 monitor 计算
center_y = 485 + 623 // 2
offset = 100
duration = 0.2

print("请确保游戏窗口已打开并处于前台")
time.sleep(1)

# 激活窗口（点击一下中心）
pyautogui.click(center_x, center_y)
time.sleep(1)

# 测试向左滑动
print("向左滑动")
pyautogui.moveTo(center_x, center_y)
pyautogui.dragRel(-offset, 0, duration=duration)
time.sleep(1)

# 测试向右滑动
print("向右滑动")
pyautogui.moveTo(center_x, center_y)
pyautogui.dragRel(offset, 0, duration=duration)
time.sleep(1)

# 测试向上滑动
print("向上滑动")
pyautogui.moveTo(center_x, center_y)
pyautogui.dragRel(0, -offset, duration=duration)
time.sleep(1)

# 测试向下滑动
print("向下滑动")
pyautogui.moveTo(center_x, center_y)
pyautogui.dragRel(0, offset, duration=duration)
time.sleep(1)

print("测试完成")