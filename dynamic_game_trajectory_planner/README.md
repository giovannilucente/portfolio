### DeepGame-TP: Dynamic Game-Theoretic Trajectory Planner
This folder contains a trajectory planner based on a game-theoretic approach. Traffic is modeled as a non-cooperative dynamic game, in which agents choose their optimal strategies while considering that other agents are also optimizing their own strategies. The trajectory planner solves the dynamic game by computing the **Generalized Nash Equilibrium (GNE)**, a state where no agent can improve its outcome unilaterally, given the strategic choices and constraints of all other agents. For further information please check the related paper: 


Giovanni Lucente, Mikkel Skov Maarssoe, Sanath Himasekhar Konthala, Anas Abulehia, Reza Dariani, Julian Schindler, **DeepGame-TP: Integrating Dynamic Game Theory and Deep Learning for Trajectory Planning**, *IEEE Open Journal of Intelligent Transportation Systems* (Volume: 5), 2024. DOI: [10.1109/OJITS.2024.3515270](https://doi.org/10.1109/OJITS.2024.3515270)


This folder contains the implementation of **DeepGame-TP** without the LSTM network presented in the paper. The application simply solves the dynamic game without using deep learning to predict the cost function.
