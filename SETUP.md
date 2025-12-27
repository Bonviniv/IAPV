# Setup do Projeto – Ambiente de Desenvolvimento

Este documento descreve a versão de Python utilizada, as bibliotecas instaladas e as instruções necessárias para trabalhar corretamente no projeto.

---

## Versão de Python

O projeto utiliza **Python 3.10.11 (64-bit)**.

Esta versão é necessária para garantir compatibilidade com as bibliotecas:

- `imitation`
- `stable-baselines3`
- `gymnasium`
- `torch`
- `pygame`

Outras versões mais recentes de Python (ex.: 3.12, 3.13, 3.14) **não são compatíveis** com algumas destas bibliotecas e não devem ser usadas neste projeto.

---

## Verificação da versão instalada

```bash
py -3.10 --version
```

Resultado esperado:

```text
Python 3.10.11
```

---

## Ambiente Virtual

É obrigatório utilizar um **ambiente virtual** para isolar as dependências do projeto.

---

### Criar o ambiente virtual

A partir da pasta raiz do projeto:

```bash
py -3.10 -m venv venv
```

---

### Ativar o ambiente virtual (Windows)

```bash
venv\Scripts\activate
```

Após ativação, o terminal deve apresentar:

```text
(venv)
```

---

### Confirmar Python ativo no ambiente virtual

```bash
python --version
```

Resultado esperado:

```text
Python 3.10.11
```

---

## Atualização das Ferramentas Base

Com o ambiente virtual ativo:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Bibliotecas Instaladas

As seguintes bibliotecas foram instaladas no ambiente virtual:

```bash
python -m pip install gymnasium stable-baselines3 imitation torch numpy matplotlib pyyaml tqdm pygame
```

### Principais dependências

- `gymnasium` – ambientes Gym
- `stable-baselines3` – algoritmos de RL (PPO)
- `imitation` – algoritmos BC e GAIL
- `torch` – redes neurais
- `pygame` – renderização e visualização
- `numpy`, `matplotlib`, `pyyaml`, `tqdm` – suporte a cálculo, gráficos e configuração

---

## Teste do Ambiente

Para verificar se todas as dependências estão corretamente instaladas, executar:

```bash
python -c "import gymnasium, stable_baselines3, imitation, pygame; print('Ambiente IAPV pronto')"
```

Resultado esperado:

```text
Ambiente IAPV pronto
```

Avisos do tipo:

```text
pkg_resources is deprecated
```

podem surgir e **não afetam o funcionamento do projeto**.

---

## Regras Importantes para Trabalhar no Projeto

- O ambiente virtual (`venv`) **deve estar sempre ativo** antes de executar qualquer script
- Não instalar bibliotecas fora do ambiente virtual
- Não utilizar versões de Python superiores à 3.10
- Confirmar sempre a presença de `(venv)` no terminal antes de trabalhar
- Todos os comandos devem ser executados a partir da pasta raiz do projeto

---

## Resumo

- Python utilizado: **3.10.11**
- Ambiente virtual: **obrigatório**
- Bibliotecas principais: `imitation`, `stable-baselines3`, `gymnasium`
- Sistema operativo testado: **Windows**
- Ambiente validado com sucesso
