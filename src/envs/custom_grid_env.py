import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomGridEnv(gym.Env):
    """
    Ambiente de Grelha Customizado para IAPV Parte 3.
    Grelha n x m com paredes aleatórias e um objetivo.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n=5, m=5, num_walls=3):
        super(CustomGridEnv, self).__init__()
        self.n = n
        self.m = m
        self.num_walls = num_walls
        self.max_steps = 50  # Limite para evitar loops infinitos
        self.current_step = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -n, -m]),
            high=np.array([n - 1, m - 1, 1, 1, 1, 1, n, m]),
            dtype=np.float32
        )

        self.agent_pos = None
        self.goal_pos = None
        self.walls = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # b/c. Escolha aleatória de objetivo e paredes
        all_coords = [(r, c) for r in range(self.n) for c in range(self.m)]
        # Garantir que temos espaço para agente, objetivo e k paredes
        chosen_indices = self.np_random.choice(
            len(all_coords), size=2 + self.num_walls, replace=False
        )
        coords = [all_coords[i] for i in chosen_indices]

        self.agent_pos = list(coords[0])
        self.goal_pos = list(coords[1])
        self.walls = coords[2:]

        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent_pos
        # Presença/ausência de paredes nas células contíguas
        walls_adj = [
            1 if (r - 1, c) in self.walls or r - 1 < 0 else 0,  # Up
            1 if (r + 1, c) in self.walls or r + 1 >= self.n else 0,  # Down
            1 if (r, c - 1) in self.walls or c - 1 < 0 else 0,  # Left
            1 if (r, c + 1) in self.walls or c + 1 >= self.m else 0  # Right
        ]
        # Posição relativa do objetivo
        rel_goal = [self.goal_pos[0] - r, self.goal_pos[1] - c]

        return np.array([r, c] + walls_adj + rel_goal, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        prev_pos = list(self.agent_pos)

        # Movimento
        if action == 0:
            self.agent_pos[0] -= 1
        elif action == 1:
            self.agent_pos[0] += 1
        elif action == 2:
            self.agent_pos[1] -= 1
        elif action == 3:
            self.agent_pos[1] += 1

        # Bloqueio por paredes ou limites
        if (tuple(self.agent_pos) in self.walls or
                not (0 <= self.agent_pos[0] < self.n and 0 <= self.agent_pos[1] < self.m)):
            self.agent_pos = prev_pos

        # LOGICA DE RECOMPENSA ALTERADA:
        terminated = self.agent_pos == self.goal_pos

        if terminated:
            reward = 60.0  # Prémio grande por ganhar
        else:
            reward = -0.5  # Pequeno custo por cada movimento (incentiva rapidez)

        # Limite de tempo (importante para o GAIL não "congelar")
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # g. A visualização do ambiente é em modo texto, no terminal
        grid = np.full((self.n, self.m), ".", dtype=str)
        for w in self.walls:
            grid[w] = "#"
        grid[tuple(self.goal_pos)] = "G"
        grid[tuple(self.agent_pos)] = "A"

        print("\n" + "\n".join([" ".join(row) for row in grid]))
        print("-" * (self.m * 2))