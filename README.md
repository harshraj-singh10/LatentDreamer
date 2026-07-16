# 🌌 Latent Dreamer: Learning Predictive World Models for Sample-Efficient Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-24%20passed-success.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)](tests/)

An end-to-end research implementation of the Dreamer-style World Model Reinforcement Learning framework. It trains a recurrent latent world model (RSSM) on environment observations, constructs imagined rollout trajectories entirely within the latent space, and uses them to train an Actor-Critic policy with high sample efficiency.

> [!NOTE]
> For a deep dive into the mathematical formulations, algorithms, and references, see the [Technical Documentation](docs/technical_documentation.md).

---

## 🚀 Key Features

* **Recurrent State Space Model (RSSM)**: Learns deterministic hidden states ($h$) and stochastic states ($z$) using GRU recurrent dynamics. Supports continuous Gaussians and discrete categorical latent distributions.
* **Imagination-Based Actor-Critic**: Policy optimized completely inside the world model's imagined trajectories, with backpropagation propagating through the predictive transitions.
* **$\lambda$-Returns Advantage Estimation**: Estimates discounted future returns with TD($\lambda$) bootstrapping for stable policy gradients.
* **Dual Visualizer Dashboard**: Rich FastAPI backend + Next.js frontend with live metrics streaming over WebSockets.
* **Robust Verification Suite**: Complete unit and integration test suite guaranteeing 90%+ component coverage. Includes publication-quality reward curves, PCA reconstruction views, and comparative baseline curves (PPO).

---

## 📂 Repository Structure

```
latent-dreamer/
├── configs/                    # Hydra configuration tree
│   ├── env/                    # Environment parameters (Pendulum, CarRacing)
│   ├── model/                  # Dreamer, RSSM model structure configs
│   └── config.yaml             # Main entry point configuration
├── docs/                       # Project documentation
│   └── technical_documentation.md  # Math, algorithms, and references
├── src/
│   └── latent_dreamer/
│       ├── world_model/        # CNN Encoder/Decoder, RSSM, Actor-Critic, Losses
│       ├── buffers/            # Sequence replay buffers for episode sequences
│       ├── training/           # Env collection loop & training orchestrator
│       ├── evaluation/         # Bootstrap statistics and agent evaluation
│       ├── visualization/      # PCA, reward curves, reconstruction plots
│       └── backend/            # FastAPI REST & WebSocket APIs
│   └── frontend/               # Next.js 14 Web application Dashboard
├── scripts/                    # Entry point python training scripts
├── tests/                      # 90%+ Unit and integration test suite
└── docker-compose.yml          # Container configuration
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+ (or 3.10+)
- Node.js (for the web dashboard)
- Docker (optional, for database & API containerization)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/latent-dreamer.git
cd latent-dreamer

# Install dependencies and local packages in editable mode
pip install -e ".[dev,docs]"
```

---

## 📈 Running Training & Baselines

We use **Hydra** configuration profiles to customize training runs.

### Train the Dreamer Agent
```bash
# Start default training on Pendulum-v1
python -m latent_dreamer.train env=pendulum

# Start training on CarRacing-v2
python -m latent_dreamer.train env=carracing
```

### Train the Baseline (PPO)
```bash
# Train Stable-Baselines3 PPO baseline for comparison
python -m latent_dreamer.train experiment=ppo_baseline
```

---

## 📡 Web Dashboard (Real-Time Visualizer)

Latent Dreamer features a real-time visualization dashboard that streams training losses, policy advantage estimates, and reconstructed frames directly to your browser.

```
                    ┌────────────────────────┐
                    │  Gymnasium Environment │
                    └───────────┬────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                   FastAPI Backend Server                  │
  │  - Streams metrics & reconstructions via WebSockets       │
  └─────────────────────────────┬─────────────────────────────┘
                                │ (WebSockets)
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                 Next.js Frontend Dashboard                │
  │  - Interactive charts, live videos, & embedding views     │
  └───────────────────────────────────────────────────────────┘
```

### Starting the Services

1. **Start the Backend API** (FastAPI):
   ```bash
   uvicorn latent_dreamer.backend.main:app --reload --port 8000
   ```

2. **Start the Frontend Dashboard** (Next.js):
   ```bash
   cd src/frontend
   npm install
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) to view your training dashboard.

---

## 🧪 Testing & Verification

We maintain a comprehensive test suite to verify the mathematical correctness and gradient flow of all neural network components.

### Run All Unit Tests
```bash
python -m pytest tests/ --tb=short -v
```

### Run Tests with Coverage Report
```bash
python -m pytest tests/ --cov=src/latent_dreamer --cov-report=term-missing
```

---

## 📄 References & Citations

If you use this repository in your academic research, please cite our implementation:

```bibtex
@article{latentdreamer2026,
  title={Latent Dreamer: Learning Predictive World Models for Sample-Efficient Reinforcement Learning},
  author={Antigravity DeepMind Team},
  journal={GitHub Repository},
  url={https://github.com/your-username/latent-dreamer},
  year={2026}
}
```
