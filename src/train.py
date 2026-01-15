import argparse
import pickle
import numpy as np
import gymnasium as gym
import torch

from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Importação do ambiente customizado
from envs.custom_grid_env import CustomGridEnv


def main():
    parser = argparse.ArgumentParser(description="Treina uma política usando BC ou GAIL")

    parser.add_argument("--file", required=True, type=str, help="Ficheiro .pkl de demonstrações")
    parser.add_argument("--output", required=True, type=str, help="Caminho para guardar a policy (.zip)")
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"], help="Ambiente")
    parser.add_argument("--algorithm", required=True, choices=["BC", "GAIL"], help="Algoritmo")

    args = parser.parse_args()

    print(f"\n--- Configuração: {args.algorithm} no ambiente {args.gym} ---")

    # 1. Carregar demonstrações
    try:
        with open(args.file, "rb") as f:
            trajectories = pickle.load(f)
        transitions = rollout.flatten_trajectories(trajectories)
        num_transitions = len(transitions)
        print(f"Sucesso: {len(trajectories)} trajetórias ({num_transitions} transições) carregadas.")
    except Exception as e:
        print(f"Erro ao carregar ficheiro: {e}")
        return

    # 2. Criar ambiente vetorizado
    def make_env():
        if args.gym == "CartPole":
            import seals
            return gym.make("seals/CartPole-v0")
        else:
            return CustomGridEnv(n=5, m=5, num_walls=3)

    env = DummyVecEnv([make_env])

    # 3. Treino
    policy = None

    if args.algorithm == "BC":
        print("A iniciar treino BC...")
        # Evita erro se houver menos de 32 transições
        b_size = min(32, num_transitions)

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(0),
            batch_size=b_size
        )
        bc_trainer.train(n_epochs=20)  # Aumentado para 20 para melhor convergência
        policy = bc_trainer.policy



    elif args.algorithm == "GAIL":

        print("A iniciar treino GAIL (isto pode demorar)...")
        d_batch = min(32, num_transitions // 2) if num_transitions > 1 else 1
        learner = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            ent_coef=0.1,
            learning_rate=0.0003,
            gamma=0.99,
            n_steps=2048

        )
        reward_net = BasicRewardNet(
            env.observation_space,
            env.action_space,
            normalize_input_layer=RunningNorm

        )

        gail_trainer = GAIL(
            demonstrations=trajectories,
            demo_batch_size=d_batch,
            gen_algo=learner,
            reward_net=reward_net,
            venv=env,
            allow_variable_horizon=True
        )

        gail_trainer.train(total_timesteps=100000)

        policy = gail_trainer.policy

    # 4. Gravar a Policy
    if policy:
        output_path = args.output if args.output.endswith(".zip") else args.output + ".zip"
        policy.save(output_path)
        print(f"\n--- SUCESSO: Policy guardada em {output_path} ---")



    env.close()


if __name__ == "__main__":
    main()