import pyautogui
print("将鼠标移到左上角，按 Ctrl+C 停止")
try:
    while True:
        x, y = pyautogui.position()
        print(f"({x}, {y})", end='\r')
except KeyboardInterrupt:
    pass