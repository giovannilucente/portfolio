### Portfolio
This repository contains some of the projects I have worked on over the years. For some of them I can share the codebase.
For more information about me and the projects I worked on, please visit my [website](https://giovannilucente.github.io/portfolio/).

## 1. DeepGame-TP: Dynamic Game-Theoretic Trajectory Planner
The topic of this project is a trajectory planner based on a game-theoretical framework. Traffic is modeled as a non-cooperative dynamic game, in which agents choose their optimal strategies while considering that other agents are also optimizing their own strategies. The trajectory planner solves the dynamic game by computing the **Generalized Nash Equilibrium (GNE)**, a state where no agent can improve its outcome unilaterally, given the strategic choices and constraints of all other agents. For further information please check the [website of the project](https://giovannilucente.github.io/portfolio/dynamic_game_trajectory_planner/index.html) and the related paper:

Giovanni Lucente, Mikkel Skov Maarssoe, Sanath Himasekhar Konthala, Anas Abulehia, Reza Dariani, Julian Schindler, **DeepGame-TP: Integrating Dynamic Game Theory and Deep Learning for Trajectory Planning**, *IEEE Open Journal of Intelligent Transportation Systems* (Volume: 5), 2024. DOI: [10.1109/OJITS.2024.3515270](https://ieeexplore.ieee.org/document/10793110).

For detailed information about the code, please refer to the [GitHub repository](https://github.com/giovannilucente/portfolio/tree/main/dynamic_game_trajectory_planner).
<p align="center">
  <img src="media/Congested_Intersection.png" alt="Congested Intersection Image" width="45%"/>
  <img src="media/Overtaking.png" alt="Overtaking" width="45%"/>
</p>

## 2. Generative AI for Trajectory Prediction
This project is currently in progress.
This repository provides code for training a model that predicts vehicle trajectories. The model takes as input an image representing the previous one-second trajectory and generates an output image depicting the predicted trajectories for the next second in the traffic scenario.
For more details, visit the [project page](https://giovannilucente.github.io/portfolio/generative_ai_trajectory_prediction/index.html) and the [GitHub repository](https://github.com/giovannilucente/portfolio/tree/main/generative_ai_trajectory_prediction).
This project is conducted in collaboration with other colleagues from DLR.
<p align="center">
  <img src="media/2_21_ground_truth.png" alt="Congested Intersection Image" width="45%"/>
  <img src="media/2_21_output.png" alt="Overtaking" width="45%"/>
</p>
Ground truth and predicted trajectories example.

## 3. Vehicle speed predictor based on LSTM-CNN hybrid neural network architecture
This repository contains code for training a model that predicts vehicle speed using a LSTM-CNN hybrid network architecture. Follow the steps below to set up and run the training process. For more information visit the [project page](https://giovannilucente.github.io/portfolio/LSTM_CNN_vehicle_speed_predictor/index.html) and the [GitHub repository](https://github.com/giovannilucente/portfolio/tree/main/LSTM_CNN_vehicle_speed_predictor).
