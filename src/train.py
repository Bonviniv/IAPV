import argparse
import pickle
import numpy as np
import gymnasium as gym
import torch
import time

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.custom_grid_env import CustomGridEnv

# Segurança para PyTorch 2.6
torch.serialization.add_safe_globals([np.ndarray, np.dtype])


def main():
    parser = argparse.ArgumentParser(description="Treino de Políticas")
    parser.add_argument("--file", type=str, help="Ficheiro .pkl de demonstrações")
    parser.add_argument("--output", required=True, type=str, help="Caminho para a policy")
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])
    parser.add_argument("--algorithm", required=True, choices=["BC", "GAIL", "PPO"])
    args = parser.parse_args()

    def make_env():
        if args.gym == "CartPole":
            import seals
            return gym.make("seals/CartPole-v0")
        else:
            return CustomGridEnv(n=5, m=5, num_walls=3)

    env = DummyVecEnv([make_env])

    # Arquitetura padrão do BC na biblioteca imitation é [32, 32]
    custom_net_arch = dict(net_arch=[32, 32])

    if args.algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        model.save(args.output)
        print(f"Sucesso: PPO guardado em {args.output}")
        return

    if not args.file:
        print("Erro: BC e GAIL exigem --file")
        return
    with open(args.file, "rb") as f:
        trajectories = pickle.load(f)

    if args.algorithm == "BC":
        print("A iniciar treino BC...")
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=np.random.default_rng(0)
        )
        bc_trainer.train(n_epochs=50)

        # AJUSTE AQUI: Criamos o PPO com a mesma arquitetura [32, 32] do BC
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=custom_net_arch)
        model.policy = bc_trainer.policy
        model.save(args.output)
        print(f"Sucesso: BC guardado com arquitetura [32, 32] em {args.output}")

    elif args.algorithm == "GAIL":
        print("A iniciar treino GAIL...")
        learner = PPO("MlpPolicy", env, verbose=1)
        reward_net = BasicRewardNet(env.observation_space, env.action_space, normalize_input_layer=RunningNorm)
        gail_trainer = GAIL(
            demonstrations=trajectories,
            demo_batch_size=32,
            gen_algo=learner,
            reward_net=reward_net,
            venv=env,
            allow_variable_horizon=True
        )
        gail_trainer.train(total_timesteps=60000)
        gail_trainer.gen_algo.save(args.output)
        print(f"Sucesso: GAIL guardado em {args.output}")

    env.close()


if __name__ == "__main__":
    main()