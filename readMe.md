# Flow Matching Labs

A hands-on lab series covering **flow matching**, **score matching**, and **generative modeling** with ODEs and SDEs. The labs progress from simulating stochastic processes to training conditional generative models on images.

## Lab Overview

| Lab | Topic | Notebook |
|-----|-------|----------|
| **Lab 1** | Simulating ODEs and SDEs | [`Lab1/lab_one.ipynb`](Lab1/lab_one.ipynb) |
| **Lab 2** | Flow Matching and Score Matching | [`Lab2/lab_two.ipynb`](Lab2/lab_two.ipynb) |
| **Lab 3** | Conditional Generative Model for Images | [`Lab3/lab_three.ipynb`](Lab3/lab_three.ipynb) |

---

## Lab 1: Simulating ODEs and SDEs

Introduces the foundations of ordinary and stochastic differential equations with numerical simulation.

**Topics covered:**
- ODE and SDE formalism (drift and diffusion coefficients)
- Euler and Euler-Maruyama numerical integration schemes
- Brownian motion: implementation and visualization
- Ornstein-Uhlenbeck process: convergence behavior and the role of θ and σ
- Langevin dynamics and score-based sampling from target distributions
- Gaussian and Gaussian mixture distributions as first-class objects

**Key questions:**
- Q1.1: Implement `EulerSimulator` and `EulerMaruyamaSimulator`
- Q2.1: Implement `BrownianMotion` SDE
- Q2.2: Implement `OUProcess` SDE
- Q3.1: Implement `LangevinSDE` for score-based sampling

---

## Lab 2: Flow Matching and Score Matching

Builds on Lab 1 to introduce conditional probability paths and the flow matching / score matching training objectives.

**Topics covered:**
- Conditional probability paths and conditional vector fields
- Gaussian conditional probability path with α_t and β_t schedules
- Linear interpolation path (α_t = t, β_t = √(1−t))
- Conditional flow matching (CFM) loss
- Score matching and the denoising score matching objective
- Training and visualizing learned vector fields and scores on 2D distributions

**Key questions:**
- Q2.1: Implement `LinearAlpha` and `SquareRootBeta`
- Q2.2: Implement `GaussianConditionalProbabilityPath.sample_conditional_path`
- Flow matching trainer and unconditional generation on 2D toy distributions

---

## Lab 3: Conditional Generative Model for Images

Scales flow matching to high-dimensional image data with conditional generation on MNIST.

**Topics covered:**
- MNIST dataset handling with `LabeledSampleable`
- Classifier-free guidance (CFG) for conditional generation
- Diffusion Transformer (DiT) architecture for parameterizing vector fields
- Conditional flow matching on 32×32 images
- Sampling and visualizing generated digits by class label

**Key questions:**
- Implementing conditional training with class labels
- Classifier-free guidance inference
- Training and sampling from the diffusion transformer
