import argparse
import gymnasium as gym
import numpy as np
import torch
import time
import os
import contextlib
from stable_baselines3 import PPO
from envs.custom_grid_env import CustomGridEnv


@contextlib.contextmanager
def torch_safe_load():
    orig_load = torch.load
    torch.load = lambda *a, **k: orig_load(*a, **{**k, 'weights_only': False})
    try:
        yield
    finally:
        torch.load = orig_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])
    args = parser.parse_args()

    if args.gym == "Custom":
        env = CustomGridEnv(n=5, m=5, num_walls=3)
    else:
        import seals
        env = gym.make("seals/CartPole-v0", render_mode="human")

    print(f"A carregar: {args.policy}")

    try:
        # O PPO.load lê a arquitetura do ficheiro zip, eliminando o size mismatch
        with torch_safe_load():
            model = PPO.load(args.policy, env=env)
        print("Sucesso: Modelo carregado com a arquitetura correta!")
    except Exception as e:
        print(f"Erro ao carregar: {e}")
        return

    obs, _ = env.reset()
    done = False
    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.3)

    os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    print("\nExecução terminada.")
    env.close()


if __name__ == "__main__":
    main()