import argparse
import gymnasium as gym
import numpy as np
import torch
import time
from stable_baselines3 import PPO
from envs.custom_grid_env import CustomGridEnv

# CORREÇÃO CRÍTICA PARA O ERRO [enforce fail]:
# Desbloqueia a leitura de ficheiros .zip do Stable Baselines no PyTorch 2.6+
torch.serialization.add_safe_globals([np.ndarray, np.dtype, torch._utils._rebuild_tensor_v2])


def main():
    parser = argparse.ArgumentParser(description="Executa uma policy (Ponto 1.b)")
    parser.add_argument("--policy", required=True, type=str, help="Caminho para a policy .zip (1.b.i)")
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"], help="Ambiente (1.b.ii)")
    args = parser.parse_args()

    # 1. Criar o Ambiente (Ponto 1.b.iii)
    if args.gym == "Custom":
        env = CustomGridEnv(n=5, m=5, num_walls=3)
    else:
        import seals
        env = gym.make("seals/CartPole-v0", render_mode="human")

    # 2. Carregar a Policy (Alínea 1.b.i)
    print(f"A carregar: {args.policy}")
    try:
        # Usamos custom_objects para garantir que o modelo carrega mesmo com avisos
        model = PPO.load(args.policy, env=env,
                         custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})
    except Exception as e:
        print(f"Erro ao carregar a policy: {e}")
        return

    # 3. Execução (Alínea 1.b.iv)
    obs, _ = env.reset()
    done = False

    # Escolha do utilizador conforme alínea 1.b.iv
    mode = input("Escolha o modo: (C)ontínuo ou (P)asso-a-passo? ").lower()

    while not done:
        # Visualização conforme ponto 1.b.iii e 2.g
        env.render()

        # Obter ação da policy P (Alínea 1.b.iv)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if mode == 'p':
            input("Pressione Enter para o próximo passo...")
        else:
            time.sleep(0.3)  # Pausa para ser visível no terminal

    if args.gym == "Custom" and terminated:
        env.render()
        print("--- OBJETIVO ATINGIDO! ---")

    env.close()


if __name__ == "__main__":
    main()