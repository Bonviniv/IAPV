Perfeito. Não preciso de mais nenhuma informação — já tenho tudo o que é necessário.
Abaixo está **UM ÚNICO BLOCO**, **UMA ÚNICA LOUSA**, **100% Markdown**, **SEM TEXTO FORA**, **SEM QUEBRAS**, pronto para copiar e colar num ficheiro `.md` (ex.: `Info/overview.md` ou similar).

````md
# Visão Geral do Projeto — IAPV Parte 3 (Aprendizagem por Imitação)

Este documento descreve o estado atual do projeto, o que já foi implementado, a organização do repositório, a função de cada componente e o que ainda falta desenvolver. O objetivo é permitir que qualquer pessoa consiga compreender rapidamente o projeto e dar-lhe seguimento.

---

## Estado Geral do Projeto

### Implementado ✔️
- Recolha de demonstrações no ambiente CartPole (`demos.py`)
- Controlo manual do agente via teclado
- Armazenamento de demonstrações em formato compatível com o pacote `imitation`
- Treino de políticas por Aprendizagem por Imitação usando Behavioral Cloning (`train.py`)
- Conversão correta de `Trajectory` → `Transitions`
- Gravação da policy treinada (`.zip`)
- Ambiente virtual configurado e dependências funcionais
- Documentação técnica em Markdown (`demos.md`, `train.md`, `testes.md`)

### Não implementado ❌ (pendente)
- Ambiente `Custom` (grid world)
- Geração de demonstrações com policy pré-treinada (`--ppo`)
- Treino com GAIL
- Execução de policy treinada (`run.py`)
- Experiências comparativas e relatório final

---

## Estrutura do Projeto

```text
IAPV_PT3/
├─ src/
│  ├─ demos.py
│  ├─ train.py
│  ├─ run.py
│  ├─ configs/
│  │  ├─ cartpole.yaml
│  │  └─ custom.yaml
│  ├─ envs/
│  │  ├─ __init__.py
│  │  └─ custom_grid_env.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ cli_utils.py
│  │  ├─ demo_utils.py
│  │  └─ eval_utils.py
│  └─ Info/
│     ├─ demos.md
│     ├─ train.md
│     └─ testes.md
├─ venv/
├─ demo.pkl
├─ policy.zip
├─ ENUNCIADO.md
├─ README.md
└─ SETUP.md
```

---

## Descrição dos Componentes

### `src/demos.py`
Responsável pela recolha de demonstrações para Aprendizagem por Imitação.

Funcionalidades:
- Recebe argumentos via linha de comando (`--gym`, `--episodes`, `--output`, `--ppo`)
- Abre o ambiente CartPole em modo de visualização
- Permite controlo manual do agente via teclado:
  - `A` → esquerda
  - `D` → direita
  - `ESC` → terminar
- Recolhe pares (estado, ação)
- Cria objetos `Trajectory` compatíveis com o pacote `imitation`
- Guarda as demonstrações num ficheiro `.pkl`

Estado:
- Totalmente funcional para CartPole
- Estrutura preparada para extensão a `Custom`

---

### `src/train.py`
Responsável pelo treino de políticas por Aprendizagem por Imitação.

Funcionalidades:
- Recebe argumentos via CLI (`--file`, `--output`, `--gym`, `--algorithm`)
- Carrega demonstrações a partir de ficheiro `.pkl`
- Converte trajectórias em `Transitions`
- Cria o ambiente Gym apropriado
- Treina uma policy usando Behavioral Cloning (BC)
- Guarda a policy treinada num ficheiro `.zip`

Estado:
- BC totalmente funcional
- Estrutura preparada para integração futura de GAIL

---

### `src/run.py`
Destinado à execução de uma policy treinada no ambiente.

Estado:
- Estrutura base criada
- Ainda não implementado nem testado

---

### `src/envs/custom_grid_env.py`
Local reservado para a implementação do ambiente Custom (grid n×m).

Estado:
- Ainda não implementado

---

### `src/utils/`
Conjunto de utilitários auxiliares para:
- parsing de argumentos
- apoio à recolha de demos
- avaliação de políticas

Estado:
- Criados como base, ainda não integrados no fluxo principal

---

## Ficheiros de Output

### `demo.pkl`
- Contém as demonstrações recolhidas
- Lista de objetos `Trajectory`
- Utilizado diretamente por `train.py`

### `policy.zip`
- Policy treinada por Behavioral Cloning
- Compatível com Stable-Baselines3
- Pronta para ser executada em `run.py`

---

## Documentação

- `ENUNCIADO.md` — enunciado oficial do trabalho
- `SETUP.md` — setup do ambiente e dependências
- `Info/demos.md` — descrição detalhada de `demos.py`
- `Info/train.md` — descrição detalhada de `train.py`
- `Info/testes.md` — como testar os scripts implementados

---




