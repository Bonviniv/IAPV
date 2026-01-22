import argparse
import gymnasium as gym
import pickle
import numpy as np
import time
import os
import pygame
from imitation.data.types import Trajectory
from stable_baselines3 import PPO
from envs.custom_grid_env import CustomGridEnv


def main():
    parser = argparse.ArgumentParser(description="Gerador de Demos IAPV")
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ppo_expert", action="store_true")
    args = parser.parse_args()

    # 1. Inicializar Ambiente
    if args.gym == "Custom":
        env = CustomGridEnv(n=5, m=5, num_walls=3)
        render_mode = "text"
    else:
        print("[INFO] A iniciar CartPole-v1...")
        env = gym.make("CartPole-v1", render_mode="human")
        # O valor original é ~0.209 radianos (12 graus)
        # 0.785 radianos = 45 graus | 1.57 radianos = 90 graus
        env.unwrapped.theta_threshold_radians = 0.785

        # Também podes aumentar o limite lateral do carrinho para ele não fugir do ecrã
        env.unwrapped.x_threshold = 4.8
        render_mode = "human"

    trajectories = []

    # 2. Gerar com PPO Expert
    if args.ppo_expert and args.gym == "CartPole":
        print("A usar PPO para gerar demonstrações...")
        model = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=15000)
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            states, actions = [obs], []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                states.append(obs);
                actions.append(action)
                done = term or trunc
            trajectories.append(Trajectory(obs=np.array(states), acts=np.array(actions), infos=None, terminal=True))
            print(f"Expert Ep {ep + 1} OK")

    # 3. Gerar Manualmente (Tu a jogar)
    else:
        if render_mode == "human":
            pygame.init()
            pygame.display.set_mode((200, 100))
            print("MODO MANUAL: Clica na janela e usa as SETAS.")

        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            states, actions = [obs], []

            print(f"\n--> Episódio {ep + 1} preparado. CLICA NA JANELA e prime ESPAÇO para começar!")

            # Pausa de segurança: espera que o utilizador prima ESPAÇO
            waiting = True
            while waiting:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        waiting = False
                    if event.type == pygame.QUIT:
                        pygame.quit();
                        return

            while not done:
                env.render()
                pygame.event.pump()

                keys = pygame.key.get_pressed()
                # 0 = Esquerda, 1 = Direita
                action = 1 if keys[pygame.K_RIGHT] else 0

                obs, reward, term, trunc, _ = env.step(action)
                states.append(obs)
                actions.append(action)
                done = term or trunc

                # Aumentamos ligeiramente o delay para conseguires reagir
                time.sleep(0.05)

            trajectories.append(Trajectory(obs=np.array(states), acts=np.array(actions), infos=None, terminal=True))
            print(f"Episódio {ep + 1} guardado com {len(actions)} passos.")

    with open(args.output, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Sucesso! Demos em: {args.output}")
    env.close()
    if render_mode == "human": pygame.quit()


if __name__ == "__main__":
    main()