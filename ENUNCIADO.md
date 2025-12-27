# Objetivos

Na Parte 3 do projeto prático os grupos terão oportunidade de explorar a utilização dos
algoritmos de aprendizagem por imitação conhecidos como Behavioral Cloning (BC) e
Generative Adversarial Imitation Learning (GAIL). A Parte 3 do projeto contribuirá para cerca
de 40% da nota final a atribuir ao Projeto.

Esta percentagem é apenas indicativa, servindo
para que o grupo possa planear o esforço de desenvolvimento, podendo ser alterada aquando
da disponibilização da restante parte do projeto.
O desenvolvimento do código será baseado no pacote Python imitation. No site desse pacote,
https://imitation.readthedocs.io/en/latest/index.html, é possível encontrar uma descrição
detalhada da sua API, assim como múltiplos exemplos de código pronto a ser executado. De
especial interesse para este projeto:

- https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
- https://imitation.readthedocs.io/en/latest/tutorials/3_train_gail.html
- https://imitation.readthedocs.io/en/latest/tutorials/10_train_custom_env.html
- https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html
- https://imitation.readthedocs.io/en/latest/main-concepts/experts.html

Com base na informação e código recolhidos da página do pacote imitation, os grupos terão
de realizar os seguintes passos para concluir a Parte 3 do projeto prático:

## 1. Criar um conjunto de ficheiros Python capazes de serem invocados a partir do terminal, recebendo um conjunto de argumentos de entrada:

### a. Criar um programa Python train.py que:

 1. Recebe o caminho para um ficheiro com um conjunto de demonstrações D;

 2. Recebe o caminho para um ficheiro de output O onde será gravado o resultado do treino;

 3. Recebe o nome do ginásio G onde D foi recolhido: CartPole (considere o "seals/CartPole-v0") ou Custom;

4. Recebe a definição de um algoritmo de aprendizagem por imitação A: BC ou GAIL;

 5. Treina uma policy P para aproximar D, utilizando A em G;

 6. Grava P no ficheiro O.

 7. Dicas:

- Exemplo de chamada no terminal: python3 train.py --file="demo_1.pkl" --gym=”CartPole” --algorithm=”GAIL” -- output=”policy.zip”

### b. Criar um programa Python run.py que:

 1. Recebe o caminho para um ficheiro de uma policy P;

 2. Recebe o nome do ginásio G onde P foi treinada: CartPole ou Custom;

 3. Abre G em modo de visualização; 

 4. Executa P em G passo a passo ou em modo contínuo, de acordo com a escolha do utilizador e enquanto o pretender.

### c. Criar um programa Python demos.py que:

 1. Recebe o nome de um ginásio G: CartPole ou Custom;

 2. Recebe um número de episódios E de demonstração;

3. Recebe o caminho para um ficheiro F onde irá gravar as demonstrações;

 4. Recebe uma flag para poder-se optar por usar uma policy pré-treinada "ppo-huggingface" para gerar o conjunto de demonstrações D (apenas para o caso CartPole);

 5. Abre G em modo de visualização;

 6. Obtém demonstrações através de policy pré-treinada (caso opção ativada) ou pedindo ao utilizador para controlar o agente em G com teclado para realizar a tarefa durante E;

 7. Armazena os pares estado-ação produzidos durante E em D;

 8. Grava D em F.

## 2. Criar um novo ginásio Custom capaz de ser usado em train.py, run.py e demos.py com as seguintes características:

 ### a. O ambiente é uma grelha regular de n x m (parâmetros do ginásio);

 ### b. Há uma célula objetivo, escolhida aleatoriamente;

 ### c. Há um conjunto de k células, escolhidas aleatoriamente, consideradas parede;

 ### d. As ações do agente são mover-se uma casa na lateral ou vertical;

 ### e. O ambiente não pode permitir que o agente atravesse paredes;

 ### f. O agente observa a sua própria posição (i.e., célula onde se encontra), a presença/ausência de paredes nas células contíguas à sua posição, e a posição relativa do objetivo em relação à posição do agente;

 ### g. A visualização do ambiente é em modo texto, no terminal;

 ### h. A tarefa do agente é chegar o mais rapidamente possível ao objetivo O.

## 3. Realizar um conjunto de experiências:

 ### a. Criar ficheiros com demonstrações recorrendo a demos.py para os ambientes CartPole e Custom, pedindo as ações ao utilizador e, no caso do CartPole, também através de policy pré-treinada;

 ### b. Treinar os agentes nos dois ambientes, utilizando as demonstrações armazenadas e recorrendo aos algoritmos BC e GAIL;

 ### c. Comparar o desempenho dos dois algoritmos de aprendizagem nos doisambientes de teste, tendo em conta múltiplas runs de treino e de execução.

## 4. Produzir um relatório que:

 ### a. Descreva detalhadamente o código desenvolvido e as experiências realizadas;

 ### b. Apresente os dados obtidos nas experiências sob forma de tabelas e gráficos;

 ### c. Inclua uma análise crítica dos resultados obtidos;

 ### d. Inclua links para vídeos produzidos pelo grupo, carregados no youtube como unlisted, onde se poderá observar os agentes já treinados em ação.