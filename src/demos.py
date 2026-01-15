import argparse
import pickle
import gymnasium as gym
import numpy as np
import torch
import seals  # Necessário para registar o namespace seals
from stable_baselines3 import PPO
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from huggingface_sb3 import load_from_hub
from envs.custom_grid_env import CustomGridEnv
from imitation.data.types import Trajectory


def main():
    parser = argparse.ArgumentParser(description="Gerar demonstrações (Ponto 1.c)")
    parser.add_argument("--gym", type=str, choices=["CartPole", "Custom"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="demos.pkl")
    parser.add_argument("--ppo", action="store_true", help="Usar ppo-huggingface (1.c.iv)")
    args = parser.parse_args()

    # 1. Ajuste de ID do ambiente para compatibilidade (Alínea 1.a.iii / 1.c.iv)
    if args.gym == "CartPole":
        # Se usarmos o PPO do HuggingFace, temos de usar o v1 padrão para o espaço de observação bater certo
        env_id = "CartPole-v1" if args.ppo else "seals/CartPole-v0"
    else:
        env_id = "CustomGridEnv-v0"

    # 2. Criação do ambiente vetorizado com RNG (Requisito imitation)
    if args.gym == "Custom":
        venv = make_vec_env(lambda: CustomGridEnv(n=5, m=5, num_walls=3),
                            n_envs=1, rng=np.random.default_rng(),
                            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)])
    else:
        venv = make_vec_env(env_id, n_envs=1, rng=np.random.default_rng(),
                            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)])

    trajectories = []

    # 3. Opção PPO Hugging Face ou Local (Alínea 1.c.iv)
    if args.ppo:
        if args.gym == "CartPole":
            print(f"--- A carregar perito do HuggingFace para {env_id} ---")
            try:
                # O repo público que já descarregaste com sucesso
                checkpoint = load_from_hub("sb3/ppo-CartPole-v1", "ppo-CartPole-v1.zip")
                expert = PPO.load(checkpoint, env=venv)
            except Exception as e:
                print(f"Erro ao carregar do Hub: {e}")
                return
        else:
            print("--- A carregar perito PPO local para Custom ---")
            try:
                expert = PPO.load("expert_ppo_custom.zip", env=venv)
            except:
                print(
                    "Erro: Treina primeiro o PPO no Custom: python src/train.py --gym Custom --algorithm PPO --output expert_ppo_custom.zip")
                return

        # Geração automática de trajetórias D (Alínea 1.c.vii)
        trajectories = rollout.generate_trajectories(
            expert, venv, rollout.make_min_episodes(args.episodes), rng=np.random.default_rng()
        )

    # 4. Modo Manual / Teclado (Alínea 1.c.vi)
    else:
        print(f"--- Modo Manual no {args.gym} (Controlos: Teclado) ---")
        for ep in range(args.episodes):
            obs, _ = venv.envs[0].reset()
            done = False
            obs_list, acts_list = [], []

            while not done:
                venv.envs[0].render()
                if args.gym == "Custom":
                    action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
                    key = input(f"[Ep {ep + 1}] Ação (w/a/s/d): ").lower()
                else:
                    action_map = {'a': 0, 'd': 1}
                    key = input(f"[Ep {ep + 1}] Ação (a: Esq, d: Dir): ").lower()

                if key in action_map:
                    action = action_map[key]
                    obs_list.append(obs)
                    acts_list.append(action)
                    obs, rew, term, trunc, _ = venv.envs[0].step(action)
                    done = term or trunc
                elif key == 'q':
                    print("Sair...")
                    return

            # Formatação obrigatória Trajectory (Alínea 1.c.viii)
            trajectories.append(Trajectory(obs=np.array(obs_list), acts=np.array(acts_list), infos=None, terminal=True))

    # 5. Gravação Final
    if trajectories:
        with open(args.output, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"\nSUCESSO: {len(trajectories)} episódios gravados em {args.output}")


if __name__ == "__main__":
    main()