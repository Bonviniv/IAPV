import seals  # TEM de vir antes do gym.make

import gymnasium as gym
import pygame
import argparse
import pickle
import numpy as np

from imitation.data.types import Trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Recolha de demonstrações para aprendizagem por imitação"
    )

    # 1. Recebe o nome de um ginásio G: CartPole ou Custom
    parser.add_argument(
        "--gym",
        required=True,
        choices=["CartPole", "Custom"],
        help="Nome do ginásio: CartPole ou Custom"
    )

    # 2. Recebe um número de episódios E de demonstração
    parser.add_argument(
        "--episodes",
        required=True,
        type=int,
        help="Número de episódios de demonstração"
    )

    # 3. Recebe o caminho para um ficheiro F onde irá gravar as demonstrações
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Caminho para o ficheiro onde as demonstrações serão guardadas"
    )

    # 4. Recebe uma flag para poder-se optar por usar uma policy pré-treinada "ppo-huggingface"
    #    para gerar o conjunto de demonstrações D (apenas para o caso CartPole);
    parser.add_argument(
        "--ppo",
        action="store_true",
        help="Usar policy pré-treinada PPO para gerar demonstrações (apenas CartPole)"
    )

    args = parser.parse_args()

    print("Ginásio:", args.gym)
    print("Episódios:", args.episodes)
    print("Ficheiro de output:", args.output)
    print("Usar PPO:", args.ppo)

    # 5. Abre G em modo de visualização (CartPole)
    if args.gym == "CartPole":
        env = gym.make("seals/CartPole-v0", render_mode="human")

     # 6. Controlo manual por teclado (1 episódio)
        obs, _ = env.reset()
        done = False

        observations = []
        actions = []

        print("Controlo: A = esquerda | D = direita | ESC = sair")

        last_action = 0
        clock = pygame.time.Clock()

        while not done:
            action = last_action

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 0
                    elif event.key == pygame.K_d:
                        action = 1
                    elif event.key == pygame.K_ESCAPE:
                        print("Execução terminada pelo utilizador.")
                        env.close()
                        return

            # 7. Armazenar par (estado, ação)
            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            last_action = action
            clock.tick(30)



        # 8. Gravar D em F (ainda dentro do if CartPole)

        observations = np.array(observations)
        actions = np.array(actions)

        # adicionar a última observação
        observations = np.vstack([observations, obs])

        infos = [{} for _ in range(len(actions))]

        traj = Trajectory(
            obs=observations,
            acts=actions,
            infos=infos,
            terminal=True
            )


        with open(args.output, "wb") as f:
            pickle.dump([traj], f)

        print(f"Demonstrações guardadas em {args.output}")

        env.close()



if __name__ == "__main__":
    main()
