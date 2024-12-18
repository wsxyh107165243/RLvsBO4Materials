# Project Overview

This repository contains codes related with "Unlocking the Black Box beyond Bayesian Global Optimization: Exploring Materials Design Spaces with Reinforcement Learning"

## Repository Structure

### BO-Daulton's
This folder contains modified code from the paper "Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization" (NeurIPS 2022). We have adapted some of the testing environments from the original implementation.

**Original Paper**: [Link to Paper](https://github.com/facebookresearch/bo_pr/blob/main/BO_Probabilistic_Reparameterization.pdf)

### BO-Ours
Contains our toy implementation of Discrete Bayesian Optimization methods.

### ModelBasedRL
Implementation of Model-based Reinforcement Learning experiments on benchmark functions:
- Ackley function
- Rastrigin function

### OnTheFlyRL
Contains experimental code comparing on-the-fly RL and discrete BO approaches on a pseudo High-Entropy Alloy (HEA) testing environment.

## HEA Testing Environment

The HEA environment data is based on the paper "A Neural Network Model for High Entropy Alloy Design" (npj Computational Materials, 2023). We have self-implemented a neural network following the authors' description to map HEA compositions to predicted values of:
- Strength
- Ductility

## Acknowledgments

This work builds upon and modifies code/data from:
- Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization (NeurIPS 2022)
- A Neural Network Model for High Entropy Alloy Design (npj Computational Materials, 2023)