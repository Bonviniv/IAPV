import argparse
import pickle
import numpy as np

import gymnasium as gym
from imitation.data import rollout

from imitation.algorithms import bc
from imitation.data.types import Transitions

from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser(
        description="Treina uma política usando BC ou GAIL"
    )

    parser.add_argument("--file", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--gym", required=True, choices=["CartPole", "Custom"])
    parser.add_argument("--algorithm", required=True, choices=["BC", "GAIL"])

    args = parser.parse_args()

    demo_file = args.file
    output_file = args.output
    gym_name = args.gym
    algorithm = args.algorithm

    print(f"Demonstrações: {demo_file}")
    print(f"Output: {output_file}")
    print(f"Ginásio: {gym_name}")
    print(f"Algoritmo: {algorithm}")

    # ============================================================
    # Fase 1 — Carregar demonstrações
    # ============================================================

    with open(demo_file, "rb") as f:
        trajectories = pickle.load(f)

    print(f"Número de trajectórias carregadas: {len(trajectories)}")

    # ============================================================
    # Fase 2 — Criar ambiente
    # ============================================================

    if gym_name == "CartPole":
        import seals

        def make_env():
            return gym.make("seals/CartPole-v0")

        env = DummyVecEnv([make_env])
    else:
        raise NotImplementedError

    # ============================================================
    # Fase 3 — Converter Trajectories para Transitions
    # ============================================================

    transitions = rollout.flatten_trajectories(trajectories)

    print("Transitions criadas:")
    print("  obs:", transitions.obs.shape)
    print("  acts:", transitions.acts.shape)
    print("  next_obs:", transitions.next_obs.shape)
    print("  dones:", transitions.dones.shape)


    # ============================================================
    # Fase 4 — Behavioral Cloning
    # ============================================================

    if algorithm == "BC":
        print("Treino com Behavioral Cloning (BC)")

        rng = np.random.default_rng(0)

        bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        )


        bc_trainer.train(n_epochs=10)
        policy = bc_trainer.policy

    # ============================================================
    # Fase 5 — Guardar policy
    # ============================================================

    policy.save(output_file)
    print(f"Policy guardada em {output_file}")


if __name__ == "__main__":
    main()
