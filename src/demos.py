import seals  # TEM de vir antes do gym.make
import gymnasium as gym
import pygame
import argparse
import pickle
import numpy as np
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO

# Importação do ambiente customizado
from envs.custom_grid_env import CustomGridEnv


def main():
    parser = argparse.ArgumentParser(description="Recolha de demonstrações para imitação")

    # 1. Nome do ginásio G: CartPole ou Custom
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])

    # 2. Número de episódios E de demonstração
    parser.add_argument("--episodes", required=True, type=int)

    # 3. Caminho para ficheiro F onde irá gravar
    parser.add_argument("--output", required=True, type=str)

    # 4. Flag para usar policy pré-treinada (apenas CartPole)
    parser.add_argument("--ppo", action="store_true")

    args = parser.parse_args()

    # Configuração do Ambiente
    if args.gym == "CartPole":
        env_id = "seals/CartPole-v0"
        env = gym.make(env_id, render_mode="human")
        venv = make_vec_env(env_id, n_envs=1)
    else:
        # Ponto 2 do enunciado: Grelha Custom n x m
        env = CustomGridEnv(n=5, m=5, num_walls=3)
        venv = None

    all_trajectories = []

    # Caso Opção PPO ativada (Apenas para CartPole)
    if args.ppo and args.gym == "CartPole":
        print("A gerar demonstrações com PPO experto...")
        expert = PPO.load("pb-gail/ppo-seals-CartPole-v0")
        all_trajectories = rollout.generate_trajectories(
            expert, venv, rollout.make_sample_until(min_episodes=args.episodes)
        )

    # Caso Controlo Manual (Teclado)
    else:
        print(f"Modo Manual: {args.episodes} episódios.")
        print("IMPORTANTE: Clica na janela do Pygame para o teclado funcionar!")
        pygame.init()
        # Cria uma pequena janela para capturar eventos de teclado
        pygame.display.set_mode((200, 200))
        clock = pygame.time.Clock()

        for e in range(args.episodes):
            obs, _ = env.reset()
            done = False
            observations, actions = [], []

            print(f"Episódio {e + 1} iniciado...")

            while not done:
                pygame.event.pump()  # Processa eventos do sistema
                action = -1

                keys = pygame.key.get_pressed()

                # Mapeamento de teclas conforme o ginásio
                if args.gym == "CartPole":
                    if keys[pygame.K_a]:
                        action = 0  # Esquerda
                    elif keys[pygame.K_d]:
                        action = 1  # Direita
                else:
                    # Custom Grid: W(Cima), S(Baixo), A(Esquerda), D(Direita)
                    if keys[pygame.K_w]:
                        action = 0
                    elif keys[pygame.K_s]:
                        action = 1
                    elif keys[pygame.K_a]:
                        action = 2
                    elif keys[pygame.K_d]:
                        action = 3

                # Só avança se uma tecla válida for premida
                if action != -1:
                    observations.append(obs)
                    actions.append(action)

                    obs, reward, terminated, truncated, _ = env.step(action)

                    if args.gym == "Custom":
                        env.render()  # Visualização em modo texto

                    done = terminated or truncated

                    # Pequeno atraso para evitar movimentos múltiplos com um clique
                    pygame.time.delay(150)

                    # Fecha se clicar no X da janela ou ESC
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        env.close()
                        return

                clock.tick(30)

            # Grava a última observação para fechar a trajetória (n+1 obs)
            observations.append(obs)
            all_trajectories.append(Trajectory(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions, dtype=np.int64),
                infos=None,
                terminal=True
            ))
            print(f"Episódio {e + 1} concluído.")

    # Gravar D em F
    with open(args.output, "wb") as f:
        pickle.dump(all_trajectories, f)
    print(f"Demonstrações guardadas em {args.output}")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()