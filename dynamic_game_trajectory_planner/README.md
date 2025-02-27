# DeepGame-TP: Dynamic Game-Theoretic Trajectory Planner
This folder contains a trajectory planner based on a game-theoretic approach. Traffic is modeled as a non-cooperative dynamic game, in which agents choose their optimal strategies while considering that other agents are also optimizing their own strategies. The trajectory planner solves the dynamic game by computing the **Generalized Nash Equilibrium (GNE)**, a state where no agent can improve its outcome unilaterally, given the strategic choices and constraints of all other agents. For further information please check the related paper: 


Giovanni Lucente, Mikkel Skov Maarssoe, Sanath Himasekhar Konthala, Anas Abulehia, Reza Dariani, Julian Schindler, **DeepGame-TP: Integrating Dynamic Game Theory and Deep Learning for Trajectory Planning**, *IEEE Open Journal of Intelligent Transportation Systems* (Volume: 5), 2024. DOI: [10.1109/OJITS.2024.3515270](https://doi.org/10.1109/OJITS.2024.3515270)


This folder contains the implementation of **DeepGame-TP** without the LSTM network presented in the paper. The application simply solves the dynamic game without using deep learning to predict the cost function. 
For further information and videos of **DeepGame-TP** implemented in [**ADORe**](https://github.com/eclipse-adore/adore), please visit this [page](https://giovannilucente.github.io/portfolio/dynamic_game_trajectory_planner/index.html).

## Installation and Setup:
To install and run this project locally, follow these steps:

### 1. Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/giovannilucente/portfolio.git
cd portfolio/dynamic_game_trajectory_planner
```

### 2. Install the requirements
Install all the required dependencies listed in the requirements.txt:
```bash
pip install -r requirements.txt
```
