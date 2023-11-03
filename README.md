# AI-Chess-Engine

Neural-network based chess engine that runs as flask web-app in users’ browser.
This project was developed to explore the concepts of convolutional neural networks and how they can be used.

## The goal of the project

The goal of the project was to learn and explore how to create, train, and use convolutional neural networks
by using one to create playable chess engine.

## Project requirements

- Chess engine evaluation function should use convolutional neural network for board state evaluation.
- Application should be hosted in AWS EC2 instance
- Application would be accessible by user using HTTPS 

## Technologies used

Project was created using Python 3.8 fallowing libraries were used:

- Python-chess – was used to simplify the creation of the engine.
- Flask – this web framework was used to enable playing chess with the computer in the browser.
- PyTorch – was used to create and train the convolutional neural network used in evaluation function.
- AWS EC2 - was used to host the application
- AWS Route 53 - was used to register domain and configure DNS 

More info in AI-Chess-Engine/AI_Chess.pdf

Link: [chess.devlake.xyz](https://chess.devlake.xyz/)
