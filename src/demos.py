import argparse
import pickle
import gymnasium as gym
import numpy as np
import os
import seals
import msvcrt
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from huggingface_sb3 import load_from_hub
from envs.custom_grid_env import CustomGridEnv
from imitation.data.types import Trajectory


def main():
    parser = argparse.ArgumentParser(description="Gerar demonstrações")
    parser.add_argument("--gym", type=str, choices=["CartPole", "Custom"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="demos.pkl")
    parser.add_argument("--ppo", action="store_true")
    args = parser.parse_args()

    # 1. Setup do Ambiente
    if args.gym == "CartPole":
        env_id = "CartPole-v1" if args.ppo else "seals/CartPole-v0"
        venv = make_vec_env(env_id, n_envs=1, rng=np.random.default_rng(),
                            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)])
    else:
        # Forçamos 1 parede apenas para garantir que há sempre caminho
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(CustomGridEnv(n=5, m=5, num_walls=1))])

    trajectories = []

    if args.ppo:
        # ... (Código PPO mantido igual)
        if args.gym == "CartPole":
            checkpoint = load_from_hub("sb3/ppo-CartPole-v1", "ppo-CartPole-v1.zip")
            expert = PPO.load(checkpoint, env=venv)
        else:
            expert = PPO.load("expert_ppo_custom.zip", env=venv)
        trajectories = rollout.generate_trajectories(expert, venv, rollout.make_min_episodes(args.episodes),
                                                     rng=np.random.default_rng())

    else:
        print(f"--- MODO MANUAL ---")
        ep_count = 0
        while ep_count < args.episodes:
            obs = venv.reset()
            done = False
            obs_list, acts_list = [obs[0]], []

            # Limpeza preventiva do estado de done
            step_count = 0

            while not done:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Episódio {ep_count + 1}/{args.episodes} | Passos: {step_count}")
                print("WASD: Mover | R: Reset Mapa | Q: Sair Forçado")
                venv.envs[0].render()

                # msvcrt.getch() bloqueia o Ctrl+C, por isso o 'q' é essencial
                char = msvcrt.getch().decode('utf-8').lower()

                if char == 'q':
                    print("A sair e a guardar o que foi feito...")
                    if trajectories:
                        with open(args.output, "wb") as f:
                            pickle.dump(trajectories, f)
                    sys.exit()

                if char == 'r':
                    break  # Sai deste loop e faz venv.reset() lá em cima

                action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3} if args.gym == "Custom" else {'a': 0, 'd': 1}

                if char in action_map:
                    action = np.array([action_map[char]])
                    obs, rew, dones, infos = venv.step(action)

                    obs_list.append(obs[0])
                    acts_list.append(action[0])
                    step_count += 1
                    done = dones[0]

            # PROTEÇÃO CONTRA EPISÓDIOS FANTASMA (Mínimo 2 passos reais)
            if step_count > 1:
                trajectories.append(
                    Trajectory(obs=np.array(obs_list), acts=np.array(acts_list), infos=None, terminal=True))
                ep_count += 1
                print(f"Episódio {ep_count} guardado!")
            else:
                print("Erro de colisão ou mapa impossível. A reiniciar episódio...")
                venv.reset()

    # Gravação
    if trajectories:
        with open(args.output, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"Sucesso: {len(trajectories)} episódios em {args.output}")


if __name__ == "__main__":
    main()