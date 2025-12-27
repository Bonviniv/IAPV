Perfeito — agora entendi exatamente o que queres 👍
Abaixo está **UM ÚNICO BLOCO**, **UMA ÚNICA LOUSA**, **TUDO DENTRO**, **SEM TEXTO FORA**, **SEM QUEBRAS**, **SEM EXPLICAÇÕES ANTES OU DEPOIS**.

Copia e cola **diretamente** para um ficheiro `.md`.

````md
# Testes — demos.py e train.py

Este documento descreve como testar os scripts `demos.py` e `train.py`, incluindo os pré-requisitos, os comandos de execução e os resultados esperados.

---

## Pré-requisitos

Antes de executar qualquer teste, é necessário garantir:

### 1. Ambiente virtual ativo

```bash
venv\Scripts\activate
```

### 2. Dependências instaladas no ambiente virtual

As seguintes bibliotecas devem estar instaladas no `venv`:

- gymnasium
- seals
- imitation
- stable-baselines3
- pygame
- numpy

### 3. Estrutura mínima do projeto

```text
IAPV_PT3/
├─ src/
│  ├─ demos.py
│  ├─ train.py
│  └─ run.py
├─ venv/
├─ README.md
├─ SETUP.md
└─ ENUNCIADO.md
```

---

## Testar demos.py

### Objetivo

Validar a recolha de demonstrações através de controlo manual, garantindo que:

- o ambiente CartPole abre corretamente em modo de visualização;
- o utilizador consegue controlar o agente via teclado;
- os pares estado-ação são recolhidos;
- as demonstrações são guardadas num ficheiro `.pkl`.

### Comando

```bash
python src/demos.py --gym CartPole --episodes 1 --output demo.pkl
```

### Durante a execução

- Abre uma janela com o ambiente CartPole.
- O agente é controlado diretamente pelo teclado:
  - `A` → ação esquerda
  - `D` → ação direita
  - `ESC` → termina a execução imediatamente
- O episódio termina automaticamente quando o ambiente sinaliza `terminated` ou `truncated`.

### Output esperado no terminal

```text
Ginásio: CartPole
Episódios: 1
Ficheiro de output: demo.pkl
Usar PPO: False
Controlo: A = esquerda | D = direita | ESC = sair
Demonstrações guardadas em demo.pkl
```

### Verificação do ficheiro gerado

```bash
dir demo.pkl
```

Deve existir um ficheiro `demo.pkl` com tamanho maior que zero.

---

## Testar train.py

### Objetivo

Treinar uma policy por Aprendizagem por Imitação utilizando Behavioral Cloning (BC), a partir das demonstrações recolhidas em `demos.py`.

### Comando

```bash
python src/train.py --file demo.pkl --gym CartPole --algorithm BC --output policy.zip
```

### Durante a execução

- O ficheiro de demonstrações é carregado.
- As trajectórias são convertidas para `Transitions`.
- O algoritmo Behavioral Cloning é treinado durante um número fixo de épocas.
- A policy treinada é guardada num ficheiro `.zip`.

### Output esperado no terminal

```text
Demonstrações: demo.pkl
Output: policy.zip
Ginásio: CartPole
Algoritmo: BC
Número de trajectórias carregadas: 1
Transitions criadas:
  obs: (N, 4)
  acts: (N,)
  next_obs: (N, 4)
  dones: (N,)
Treino com Behavioral Cloning (BC)
Policy guardada em policy.zip
```

(`N` depende do número de passos recolhidos durante a demonstração)

### Verificação do ficheiro gerado

```bash
dir policy.zip
```

Deve existir um ficheiro `policy.zip`.

---

## Estado atual do projeto

- `demos.py`: funcional para CartPole com controlo manual.
- `train.py`: funcional para Behavioral Cloning (BC).
- `Custom`: ainda não implementado.
- `GAIL`: ainda não implementado.
- `run.py`: ainda não testado.

Este conjunto de testes confirma que a pipeline básica de:
**recolha de demonstrações → treino por BC → gravação da policy**
está corretamente implementada.
````
