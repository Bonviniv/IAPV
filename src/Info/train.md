# train.py — Treino de Policies por Aprendizagem por Imitação

Este ficheiro implementa o programa `train.py`, responsável por treinar uma policy a partir de demonstrações recolhidas, utilizando algoritmos de aprendizagem por imitação, conforme definido no enunciado da Parte 3 do projeto de IAPV.

Nesta fase, o foco está no algoritmo **Behavioral Cloning (BC)** e no ambiente **CartPole**.

---

## Objetivo

Receber um conjunto de demonstrações **D**, um ambiente **G** e um algoritmo de aprendizagem por imitação **A**, treinar uma policy **P** que aproxime o comportamento demonstrado e gravar essa policy num ficheiro de output.

---

## Interface de Linha de Comandos (CLI)

O programa aceita os seguintes argumentos:

- `--file D`  
  Caminho para o ficheiro `.pkl` contendo as demonstrações (lista de `Trajectory`).

- `--output O`  
  Caminho onde a policy treinada será guardada (ficheiro `.zip`).

- `--gym {CartPole, Custom}`  
  Nome do ambiente onde as demonstrações foram recolhidas.

- `--algorithm {BC, GAIL}`  
  Algoritmo de aprendizagem por imitação a utilizar.

A interface está diretamente alinhada com o enunciado.

---

## Funcionamento Geral

### Fase 1 — Carregamento das Demonstrações

- O ficheiro indicado por `--file` é carregado com `pickle`.
- Espera-se uma lista de objetos `Trajectory`.
- É feito um print informativo com o número de trajectórias carregadas.

---

### Fase 2 — Criação do Ambiente Gym

- Para o caso `CartPole`:
  - o ambiente `seals/CartPole-v0` é criado,
  - encapsulado num `DummyVecEnv`, como esperado pela API do `imitation`.
- O módulo `seals` é importado explicitamente para garantir o registo do ambiente.

O ambiente `Custom` ainda não está implementado.

---

### Fase 3 — Conversão para Transitions

- As trajectórias são convertidas para `Transitions` através de  
  `rollout.flatten_trajectories`.
- Este formato é o esperado pelos algoritmos BC e GAIL.
- São mostradas as dimensões dos dados para verificação.

---

### Fase 4 — Treino com Behavioral Cloning (BC)

- Quando `--algorithm BC` é selecionado:
  - é criado um gerador de números aleatórios (`rng`) para reprodutibilidade,
  - é inicializado o treinador `bc.BC`,
  - a policy é treinada durante um número fixo de épocas (`n_epochs=10`).

No final do treino, a policy aprendida é extraída.

---

### Fase 5 — Gravação da Policy

- A policy treinada é guardada no caminho indicado por `--output`,
- O formato gerado (`.zip`) é compatível com `stable-baselines3` e `run.py`.

---

## Decisões Técnicas Relevantes

- **Uso de `DummyVecEnv`**:  
  Necessário para compatibilidade com a API do `imitation`.

- **Uso de `rollout.flatten_trajectories`**:  
  Método estável e compatível com a versão atual da biblioteca.

- **Uso explícito de `rng`**:  
  Exigido pela API do `imitation` para garantir reprodutibilidade.

---

## O que já está completo

- Interface CLI conforme o enunciado
- Carregamento correto de demonstrações
- Criação do ambiente CartPole
- Conversão para `Transitions`
- Treino funcional com Behavioral Cloning
- Gravação correta da policy treinada

---

## O que ainda falta implementar

1. **Algoritmo GAIL**
2. **Ambiente Custom**
3. Ajustes experimentais:
   - número de épocas
   - comparação entre BC e GAIL
   - múltiplas runs

---

## Estado Atual

O `train.py` encontra-se **funcional, estável e alinhado com o enunciado**, permitindo já treinar policies por aprendizagem por imitação e servir de base direta para a execução (`run.py`) e para as experiências pedidas no relatório.
