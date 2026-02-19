import numpy as np
import pickle
import argparse
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL
from d3rlpy.models.encoders import NatureCNNEncoderFactory
from d3rlpy.metrics import td_error_scorer, initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split
import gymnasium as gym
from subway_surfers_env import SubwaySurfersEnv  # 你的环境类

def load_from_pickles(file_list):
    """从多个 pickle 文件加载数据，返回 MDPDataset"""
    observations = []
    actions = []
    rewards = []
    terminals = []

    for fname in file_list:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        for t in data:
            # 扁平化图像 (120,160,3) -> (57600,)
            obs_flat = t['obs'].flatten().astype(np.float32) / 255.0
            observations.append(obs_flat)
            actions.append(t['action'])
            rewards.append(t['reward'])
            terminals.append(t['done'])

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
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='subway_dataset.h5',
                        help='Path to HDF5 dataset or list of pickle files (comma separated)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Save model every N steps')
    args = parser.parse_args()

    # 加载数据集
    if args.dataset.endswith('.h5'):
        dataset = MDPDataset.load(args.dataset)
    else:
        # 假设传入的是用逗号分隔的pickle文件列表
        file_list = args.dataset.split(',')
        dataset = load_from_pickles(file_list)

    print(f"数据集加载完成：{dataset}")

    # 划分训练/验证集（可选）
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    # 创建CQL算法实例
    # 使用Nature CNN编码器处理图像
    encoder_factory = NatureCNNEncoderFactory()
    cql = CQL(
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        batch_size=args.batch_size,
        n_frames=1,               # 不使用帧堆叠
        n_steps=args.total_steps,
        n_steps_per_epoch=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        use_gpu=args.gpu if args.gpu >= 0 else False,
        discrete_action=True,
        # CQL特有参数
        conservative_weight=5.0,   # 保守系数，可根据需要调整
    )

    # 定义评估环境（用于训练中监控，可选）
    eval_env = SubwaySurfersEnv(render_mode=None)

    # 开始训练
    cql.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.total_steps,
        n_steps_per_epoch=1000,
        scorers={
            'td_error': td_error_scorer,
            'value': initial_state_value_estimation_scorer,
            'environment': evaluate_on_environment(eval_env)  # 会在真实环境中评估，注意这会很慢
        },
        save_interval=args.save_interval,  # 每 save_interval 步保存一次模型
        tensorboard_dir='./cql_tensorboard',
        verbose=True,
        show_progress=True,
    )

    # 保存最终模型
    cql.save_model('cql_subway_final.pt')
    print("训练完成，模型已保存")

if __name__ == '__main__':
    main()