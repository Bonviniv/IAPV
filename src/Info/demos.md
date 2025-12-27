# demos.py — Recolha de Demonstrações

Este ficheiro implementa o programa `demos.py`, responsável por recolher demonstrações (estado–ação) para aprendizagem por imitação, de acordo com o enunciado da Parte 3 do projeto de IAPV.

---

## Objetivo

Permitir a criação de um conjunto de demonstrações **D** num ambiente Gym, através de:
- controlo manual do utilizador (teclado),
- com visualização do ambiente,
- e gravação das demonstrações num ficheiro para posterior treino com BC ou GAIL.

Nesta fase, o foco está no ambiente **CartPole**.

---

## Interface de Linha de Comandos (CLI)

O programa recebe os seguintes argumentos:

- `--gym {CartPole, Custom}`  
  Define o ambiente onde as demonstrações são recolhidas.

- `--episodes E`  
  Número de episódios de demonstração (atualmente ainda não utilizado no ciclo principal).

- `--output F`  
  Caminho do ficheiro onde as demonstrações são guardadas (`.pkl`).

- `--ppo`  
  Flag para usar uma policy pré-treinada PPO (prevista no enunciado, ainda não implementada).

Esta interface está totalmente alinhada com o enunciado.

---

## Funcionamento Geral

### 1. Criação do Ambiente

- Para o caso `CartPole`, o ambiente é criado com:
  - `seals/CartPole-v0`
  - `render_mode="human"` para visualização.
- O módulo `seals` é importado explicitamente para garantir o registo correto do ambiente.

---

### 2. Controlo Manual do Agente

- O controlo é feito via **teclado**, usando `pygame`.
- Mapeamento de teclas:
  - `A` → ação esquerda
  - `D` → ação direita
  - `ESC` → termina a execução imediatamente
- O loop avança continuamente a uma taxa limitada (`clock.tick(30)`), sem bloquear à espera de input textual.

Durante a execução:
- o último estado (`obs`) e a ação escolhida são armazenados a cada passo.

---

### 3. Recolha das Demonstrações

- São guardados:
  - uma lista de observações (`obs`)
  - uma lista de ações (`acts`)
- No final do episódio:
  - a última observação é adicionada,
  - é criado um objeto `Trajectory` compatível com a biblioteca `imitation`.

Formato final:
- uma lista contendo uma `Trajectory`,
- serializada com `pickle` no ficheiro indicado por `--output`.

---

## Decisões Técnicas Importantes

- **Uso de `Trajectory`**:  
  Garante compatibilidade direta com `imitation` e com o processo de treino em `train.py`.

- **Uso de `pygame`**:  
  Permite controlo em tempo real com foco na janela gráfica, evitando interação pelo terminal.

- **Execução por episódio único**:  
  Simplifica a validação inicial do pipeline (demos → treino → execução).

---

## O que já está completo

- Interface CLI conforme o enunciado
- Abertura do ambiente em modo visualização
- Controlo manual por teclado
- Recolha de pares estado–ação
- Criação de `Trajectory` válida
- Gravação das demonstrações em ficheiro `.pkl`
- Integração funcional com `train.py`

---

## O que ainda falta implementar

De forma explícita e consciente:

1. **Suporte a múltiplos episódios (`--episodes`)**  
   Atualmente o código executa apenas um episódio; falta um ciclo externo sobre `E`.

2. **Geração automática de demonstrações com PPO (`--ppo`)**  
   A flag existe, mas ainda não carrega nem executa uma policy pré-treinada.

3. **Suporte ao ambiente `Custom`**  
   Previsto no enunciado, será implementado numa fase posterior do projeto.

---

## Estado Atual

O `demos.py` encontra-se num estado **funcional, estável e aceitável** para o projeto nesta fase, permitindo já:
- gerar demonstrações válidas,
- treinar policies com BC,
- e validar o pipeline completo de aprendizagem por imitação.

As funcionalidades em falta são extensões naturais e não bloqueiam o progresso do trabalho.
