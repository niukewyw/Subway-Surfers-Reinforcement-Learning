````
**The game executable (.exe) used in this project is sourced from: https://github.com/vinaymancha/Subway-Surfers-AI**

# **Subway Surfers Reinforcement Learning Project**

## Abstract

This project encapsulates the game Subway Surfers as a standard Gymnasium environment using computer vision and input simulation techniques. It supports both online reinforcement learning (PPO) and offline reinforcement learning (CQL). The project includes complete code and tools to implement the full pipeline from environment calibration and data collection to model training and evaluation.

---

## Environment Dependencies

Install the required Python libraries:

```bash
pip install gymnasium opencv-python pyautogui mss numpy torch torchvision stable-baselines3 d3rlpy easyocr pynput
```

Make sure the game executable `Subway_Surfers.exe` is located in the project directory (or modify the path in the code accordingly).

---

## File Description

| File Name | Description |
|------------|-------------|
| `subway_surfers_env.py` | Custom Gymnasium environment that wraps game interaction, OCR recognition, auto-restart, and anti-stuck mechanisms |
| `test_location.py` | Used to measure screen coordinates (e.g., mileage, coins, button positions); outputs real-time mouse coordinates |
| `test_swipe.py` | Tests whether mouse swipe actions are correctly recognized by the game; verifies `pyautogui` swipe parameters |
| `test_env.py` | Tests environment stability using random actions; checks auto-restart and reward calculation |
| `collect_random.py` | Collects random policy data and saves it as pickle files every `save_interval` steps |
| `collect_human.py` | Records human demonstrations using keyboard arrow keys to generate high-quality trajectories |
| `merge_random_data_and_human_data.py` | Merges random and human data and converts them into d3rlpy-compatible HDF5 format |
| `ppo_train.py` | Performs online training using PPO from Stable-Baselines3; saves the model every 100k steps |
| `test_ppo_model.py` | Loads a trained PPO model and runs it in the real game |
| `offline_train.py` | Performs offline training using CQL from d3rlpy based on collected datasets |
| `evaluate_offline_rl.py` | Loads a trained CQL model and evaluates it in the real game |
| `offline_example` | Example description file for offline training datasets |
| `Subway_Surfers.exe` | Game executable file (must be provided manually) |
| `Subway_Surfers_Data/` | Required data folder for the game (must be in the same directory as the exe) |

---

# Step 1: Environment Calibration

Before starting any training, you must calibrate the game window position and OCR recognition regions.

## Get Screen Coordinates

Run `test_location.py`. Move the mouse to the positions you want to measure, and the script will output the current coordinates in real time. Record the top-left and bottom-right coordinates of the following regions:

- **Mileage area** (first row of numbers at the top-right during gameplay)
- **Coin area** (second row of numbers at the top-right)
- **Death score area** (center score display after game over)
- **Main menu PLAY button** (appears when first launching the game)
- **Start button** (appears after clicking PLAY but before gameplay starts)
- **Death screen PLAY button** (replay button after game over)

```bash
python test_location.py
```

Move the mouse to the target position and press `Ctrl+C` to stop the script and record the coordinates.

---

## Modify Coordinates in the Environment

Open `subway_surfers_env.py`, locate the `__init__` method, and replace the coordinates with your measured values:

```python
# Mileage region: top-left (x1,y1), bottom-right (x2,y2)
self.mileage_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# Coin region
self.coin_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# Death score region
self.score_region = {"left": x1, "top": y1, "width": x2-x1, "height": y2-y1}
# Button coordinates
self.play_button_center_death = (x, y)   # Death screen PLAY button center
self.play_button_center_main = (x, y)    # Main menu PLAY button center
self.start_button_center = (x, y)        # Start button center
```

---

## Test Mouse Swipes

Run `test_swipe.py` and observe whether the character correctly responds to left, right, up, and down swipes. If swipes are ineffective, adjust `swipe_offset` (distance) and `swipe_duration` (duration) in `subway_surfers_env.py`.

```bash
python test_swipe.py
```

---

# Step 2: Test the Environment

Run `test_env.py` to interact with the game using random actions and verify that the environment runs stably:

- Can the game launch automatically and enter a round?
- Can it automatically restart after death?
- Can OCR correctly read mileage and coins?
- Is reward calculation reasonable?

```bash
python test_env.py
```

Check the console output and ensure there are no errors. If issues occur, adjust coordinates or waiting times accordingly.

---

# Step 3: Data Collection (for Offline Training)

Offline training requires collecting a large amount of trajectory data. Two collection methods are provided.

## Collect Random Policy Data

Run `collect_random.py` to interact with the environment using random actions. Data will be saved as pickle files (e.g., `random_data_10000.pkl`). By default, 80,000 steps are collected, saving every 10,000 steps.

```bash
python collect_random.py
```

---

## Collect Human Demonstration Data

Run `collect_human.py`. Use the keyboard arrow keys (left/right/up/down) to control the character. Releasing a key stops the action. Press `ESC` to exit and save the data as `manual_data.pkl`.

```bash
python collect_human.py
```

It is recommended to collect around 20,000 steps of human data to improve dataset quality.

---

## Merge Datasets

After collecting random and human data, run `merge_random_data_and_human_data.py` to merge pickle files and convert them into d3rlpy-compatible HDF5 format (e.g., `subway_dataset.h5`).

```bash
python merge_random_data_and_human_data.py
```

Specify the input file list and output filename inside the script.

---

# Step 4: Online Training (Optional)

To quickly validate the environment, you can use `ppo_train.py` for online training. This script uses PPO from Stable-Baselines3, trains for 1,000,000 steps, and saves the model every 100,000 steps in the `./ppo_models/` directory.

```bash
python ppo_train.py
```

Use TensorBoard to monitor training:

```bash
tensorboard --logdir ./ppo_tensorboard
```

After training, test the model using:

```bash
python test_ppo_model.py --model_path ./ppo_models/subway_ppo_100000.zip
```

---

# Step 5: Offline Training

Use the collected dataset for offline training by running `offline_train.py`. This script uses the CQL algorithm from d3rlpy and requires specifying the dataset path (e.g., `subway_dataset.h5`).

```bash
python offline_train.py --dataset subway_dataset.h5 --total_steps 200000 --gpu 0
```

Training parameter explanation:

- `--dataset`: Path to the HDF5 dataset or a comma-separated list of pickle files
- `--total_steps`: Number of training update steps (not environment steps)
- `--gpu`: Specify GPU ID; use -1 for CPU
- `--batch_size`: Batch size (default 32)
- `--save_interval`: Model save interval (default 10000 steps)

Training logs will be generated in `./cql_tensorboard`, which can be viewed using TensorBoard.

---

## Evaluate Offline Model

After training, evaluate the model in the real game using:

```bash
python evaluate_offline_rl.py --model_path cql_subway_final.pt --n_episodes 5
```

The script will open the game window and display the agent’s performance.

---

# Frequently Asked Questions

## What if OCR recognition is inaccurate?

- Adjust the binarization threshold (`cv2.threshold(gray, 200, 255, ...)` in `_ocr_digit` can be changed to 150 or 180).
- Ensure the coordinate regions precisely contain only digits and not background.
- Consider replacing EasyOCR with `pytesseract` or `paddleocr`.

---

## What if the game window loses focus and inputs fail?

- Ensure the game window remains in the foreground and is not minimized or covered by other windows.
- Avoid performing other operations during training.

---

## How does the anti-stuck mechanism work?

If the mileage remains unchanged for `stuck_threshold` consecutive steps (default 10), the environment enters “stuck mode,” forcing the action to 0. If stuck mode continues for more than `stuck_terminate_threshold` steps (default 30), the episode is forcibly terminated. On the next `reset`, the game process will be killed and restarted to ensure full recovery.

---

# Contribution

Issues and pull requests are welcome!

---

**The project code is generated by DeepSeek. The training performance is currently unknown and still under testing.**
````
