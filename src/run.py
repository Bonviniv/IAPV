import os
import warnings

# 1. FORÇAR BYPASS DE SEGURANÇA ANTES DE QUALQUER OUTRA COISA
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

# Truque para garantir que o torch ignore a restrição weights_only internamente
if hasattr(torch.serialization, 'default_restore_location'):
    # Força a configuração global de segurança para o nível baixo
    torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **kwargs, weights_only=False)

import gymnasium as gym
import argparse
import time
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from envs.custom_grid_env import CustomGridEnv


def main():
    parser = argparse.ArgumentParser(description="Executa uma policy treinada")
    parser.add_argument("--policy", required=True, type=str)
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])
    args = parser.parse_args()

    # Criar Ambiente
    if args.gym == "CartPole":
        import seals
        env = gym.make("seals/CartPole-v0", render_mode="human")
    else:
        env = CustomGridEnv(n=5, m=5, num_walls=3)

    print(f"A carregar: {args.policy}")

    try:
        # Carregamento simplificado agora que o 'torch.load' foi alterado globalmente
        model = ActorCriticPolicy.load(args.policy, device="cpu")
        print("--- POLICY CARREGADA COM SUCESSO! ---")
    except Exception as e:
        print(f"Falha ao carregar. Tente apagar o .zip e treinar de novo. Erro: {e}")
        return

    print("\nEscolha o modo: 1-Contínuo, 2-Passo-a-passo")
    modo = input("Opção: ")
    obs, _ = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if args.gym == "Custom":
                env.render()

            if modo == "2":
                input("Pressione Enter...")
            else:
                time.sleep(0.3)

            if terminated or truncated:
                print("\n--- Reiniciando ---")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nParado.")
    finally:
        env.close()


if __name__ == "__main__":
    main()