**本项目所使用的游戏可执行文件（.exe）来源于：https://github.com/vinaymancha/Subway-Surfers-AI ，可在本地直接运行。项目代码由 DeepSeek 生成。目前训练效果尚不明确，仍在测试阶段。但可以确定的一点是，由于无法直接获取游戏的源数据，只能通过屏幕识别进行离线采集和在线训练，因此训练与数据收集过程将会非常缓慢。此外，在在线训练和离线数据采集过程中，会占用你的电脑鼠标，导致电脑无法执行其他任务。因此，本项目仅供娱乐用途。如果未来能够训练出更有效的模型并收集到完整的离线数据集，我会及时上传。**

# **Subway Surfers 强化学习项目**

## 摘要

本项目通过计算机视觉与输入模拟技术，将 Subway Surfers 游戏封装为一个标准的 Gymnasium 环境。项目同时支持在线强化学习（PPO）与离线强化学习（CQL）。包含从环境校准、数据采集到模型训练与评估的完整流程代码与工具。

---

## 环境依赖

安装所需的 Python 库：

```bash
pip install gymnasium opencv-python pyautogui mss numpy torch torchvision stable-baselines3 d3rlpy easyocr pynput
```

请确保游戏可执行文件 `Subway_Surfers.exe` 位于项目目录中（或在代码中修改对应路径）。

---

## 文件说明

| 文件名 | 说明 |
|------------|-------------|
| `subway_surfers_env.py` | 自定义 Gymnasium 环境，封装游戏交互、OCR识别、自动重启与防卡死机制 |
| `test_location.py` | 用于测量屏幕坐标（如里程、金币、按钮位置）；实时输出鼠标坐标 |
| `test_swipe.py` | 测试鼠标滑动动作是否被游戏正确识别；验证 `pyautogui` 参数 |
| `test_env.py` | 使用随机动作测试环境稳定性；检查自动重启与奖励计算 |
| `collect_random.py` | 使用随机策略采集数据，每隔 `save_interval` 步保存为 pickle 文件 |
| `collect_human.py` | 使用键盘方向键记录人工演示轨迹，生成高质量数据 |
| `merge_random_data_and_human_data.py` | 合并随机数据与人工数据，并转换为 d3rlpy 可用的 HDF5 格式 |
| `ppo_train.py` | 使用 Stable-Baselines3 中的 PPO 进行在线训练；每 100k 步保存一次模型 |
| `test_ppo_model.py` | 加载已训练的 PPO 模型并在真实游戏中运行 |
| `offline_train.py` | 使用 d3rlpy 中的 CQL 算法进行离线训练 |
| `evaluate_offline_rl.py` | 加载已训练的 CQL 模型并在真实游戏中评估 |
| `offline_example` | 离线训练数据示例说明文件 |
| `Subway_Surfers.exe` | 游戏可执行文件（需手动提供） |
| `Subway_Surfers_Data/` | 游戏所需数据文件夹（需与 exe 位于同一目录） |

---

# 第一步：环境校准

在开始训练之前，必须校准游戏窗口位置与 OCR 识别区域。

---

## 获取屏幕坐标

运行 `test_location.py`。将鼠标移动到需要测量的位置，脚本会实时输出当前坐标。记录以下区域的左上角与右下角坐标：

- **里程区域**（游戏过程中右上角第一行数字）
- **金币区域**（右上角第二行数字）
- **死亡得分区域**（游戏结束后的中央得分显示）
- **主菜单 PLAY 按钮**
- **Start 按钮**（点击 PLAY 后、游戏正式开始前）
- **死亡界面 PLAY 按钮**

```bash
python test_location.py
```

将鼠标移动到目标位置，按 `Ctrl+C` 退出并记录坐标。

---

## 修改环境中的坐标

打开 `subway_surfers_env.py`，在 `__init__` 方法中替换为你测量的坐标：

```python
# 里程区域：左上 (x1,y1)，右下 (x2,y2)
self.mileage_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# 金币区域
self.coin_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# 死亡得分区域
self.score_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# 按钮坐标
self.play_button_center_death = (x, y)   # 死亡界面 PLAY 按钮中心
self.play_button_center_main = (x, y)    # 主菜单 PLAY 按钮中心
self.start_button_center = (x, y)        # Start 按钮中心
```

---

## 测试鼠标滑动

运行 `test_swipe.py`，观察角色是否正确响应左右上下滑动。如果无效，请在 `subway_surfers_env.py` 中调整：

- `swipe_offset`（滑动距离）
- `swipe_duration`（滑动时间）

```bash
python test_swipe.py
```

---

# 第二步：测试环境

运行 `test_env.py`，使用随机动作与游戏交互，验证环境稳定性：

- 是否能自动启动游戏并进入对局？
- 是否能在死亡后自动重启？
- OCR 是否能正确读取里程与金币？
- 奖励计算是否合理？

```bash
python test_env.py
```

检查控制台输出，确保无报错。如有问题，请调整坐标或等待时间参数。

---

# 第三步：数据采集（用于离线训练）

离线训练需要大量轨迹数据。提供两种采集方式。

---

## 采集随机策略数据

运行 `collect_random.py`。默认采集 80,000 步，每 10,000 步保存为一个 pickle 文件（如 `random_data_10000.pkl`）。

```bash
python collect_random.py
```

---

## 采集人工演示数据

运行 `collect_human.py`，使用键盘方向键控制角色（左右上下）。松开按键即停止动作。按 `ESC` 退出并保存为 `manual_data.pkl`。

```bash
python collect_human.py
```

建议采集约 20,000 步人工数据以提升数据质量。

---

## 合并数据集

运行 `merge_random_data_and_human_data.py`，将多个 pickle 文件合并并转换为 d3rlpy 可用的 HDF5 格式（如 `subway_dataset.h5`）。

```bash
python merge_random_data_and_human_data.py
```

在脚本中指定输入文件列表与输出文件名。

---

# 第四步：在线训练（可选）

可使用 `ppo_train.py` 快速验证环境。该脚本使用 Stable-Baselines3 的 PPO 算法训练 1,000,000 步，并每 100,000 步保存一次模型到 `./ppo_models/`。

```bash
python ppo_train.py
```

使用 TensorBoard 监控训练：

```bash
tensorboard --logdir ./ppo_tensorboard
```

训练完成后测试模型：

```bash
python test_ppo_model.py --model_path ./ppo_models/subway_ppo_100000.zip
```

---

# 第五步：离线训练

运行 `offline_train.py` 使用 CQL 算法进行离线训练：

```bash
python offline_train.py --dataset subway_dataset.h5 --total_steps 200000 --gpu 0
```

参数说明：

- `--dataset`：HDF5 数据集路径或逗号分隔的 pickle 文件列表
- `--total_steps`：训练更新步数（非环境交互步数）
- `--gpu`：GPU 编号；使用 CPU 请设置为 -1
- `--batch_size`：批大小（默认 32）
- `--save_interval`：模型保存间隔（默认 10000 步）

训练日志保存在 `./cql_tensorboard`，可使用 TensorBoard 查看。

---

## 评估离线模型

训练完成后运行：

```bash
python evaluate_offline_rl.py --model_path cql_subway_final.pt --n_episodes 5
```

脚本会自动打开游戏窗口并展示智能体表现。

---

# 贡献

欢迎提交 Issue 与 Pull Request！
