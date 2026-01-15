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

from envs.custom_grid_env import CustomGridEnv

# Correção de Segurança para carregar modelos no PyTorch 2.6+
torch.serialization.add_safe_globals([np.ndarray, np.dtype])

def main():
    parser = argparse.ArgumentParser(description="Treino de Políticas (BC, GAIL, PPO)")

    # Tornamos o --file opcional apenas para o PPO
    parser.add_argument("--file", type=str, help="Ficheiro .pkl de demonstrações D")
    parser.add_argument("--output", required=True, type=str, help="Caminho para a policy O")
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"], help="Ambiente G")
    # ADICIONADO 'PPO' às escolhas permitidas
    parser.add_argument("--algorithm", required=True, choices=["BC", "GAIL", "PPO"], help="Algoritmo A")

    args = parser.parse_args()

    # 1. Configuração do Ambiente
    def make_env():
        if args.gym == "CartPole":
            import seals
            return gym.make("seals/CartPole-v0")
        else:
            return CustomGridEnv(n=5, m=5, num_walls=3)

    env = DummyVecEnv([make_env])

    # 2. Lógica específica para PPO (Treino do Perito)
    if args.algorithm == "PPO":
        print(f"\n--- Treinando PPO experto no ambiente {args.gym} ---")
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1, learning_rate=0.0003)
        model.learn(total_timesteps=100000)
        model.save(args.output)
        print(f"Sucesso: Modelo guardado em {args.output}")
        return

    # 3. Carregar demonstrações para BC e GAIL
    if not args.file:
        print("Erro: BC e GAIL exigem um ficheiro de demonstrações (--file).")
        return

    try:
        with open(args.file, "rb") as f:
            trajectories = pickle.load(f)
        print(f"Sucesso: {len(trajectories)} trajetórias carregadas.")
    except Exception as e:
        print(f"Erro ao carregar ficheiro: {e}")
        return

    # 4. Treino de Imitação
    policy = None
    if args.algorithm == "BC":
        print("A iniciar treino BC...")
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=np.random.default_rng(0)
        )
        bc_trainer.train(n_epochs=30)
        policy = bc_trainer.policy

    elif args.algorithm == "GAIL":
        print("A iniciar treino GAIL...")
        learner = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
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
        policy = gail_trainer.policy

    # 5. Gravação Final
    if policy:
        policy.save(args.output)
        print(f"Policy guardada em {args.output}")

    env.close()

if __name__ == "__main__":
    main()