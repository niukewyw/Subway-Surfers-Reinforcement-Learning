import pickle
import numpy as np
from d3rlpy.dataset import MDPDataset

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def merge_and_convert(random_file, manual_file, output_h5):
    random_data = load_pickle(random_file)
    manual_data = load_pickle(manual_file)
    all_data = random_data + manual_data
    print(f"总数据步数: {len(all_data)}")

    observations = []
    actions = []
    rewards = []
    terminals = []

    for t in all_data:
        # 扁平化并归一化
        obs_flat = t["obs"].flatten().astype(np.float32) / 255.0
        observations.append(obs_flat)
        actions.append(t["action"])
        rewards.append(t["reward"])
        terminals.append(t["done"])

    observations = np.array(observations)
    actions = np.array(actions).reshape(-1, 1)
    rewards = np.array(rewards)
    terminals = np.array(terminals, dtype=bool)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        discrete_action=True
    )
    dataset.dump(output_h5)
    print(f"数据集已保存至 {output_h5}")

if __name__ == "__main__":
    merge_and_convert("random_data_final.pkl", "manual_data.pkl", "subway_10w.h5")