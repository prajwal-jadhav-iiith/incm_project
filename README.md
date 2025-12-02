# From Q-Values to Cognitive Maps

**Emergence of Place Cell Representations in a Reinforcement Learning Agent**

This project investigates the emergence of spatial representations (specifically "place cells") in artificial neural networks. By training a simple feedforward network to decode spatial coordinates from the internal state (Q-values) of a Reinforcement Learning agent, we demonstrate that biological-like spatial tuning can arise spontaneously from task-optimized representations.

**Authors:**
*   Jadhav Prajwal Dadaram (Roll no: 2025701045)
*   Mafruz Nissar Rahman (Roll no: 2025701024)

---

## 🚀 Project Overview

The project is divided into three distinct phases:

1.  **Phase 1: Reinforcement Learning**
    *   A tabular Q-learning agent is trained to navigate various 2D gridworld environments (Open Field, T-Maze, Four Rooms, etc.).
    *   The agent learns to reach a goal location efficiently, building a set of Q-values that implicitly contain spatial structure.

2.  **Phase 2: Spatial Decoder**
    *   A feedforward neural network is trained to predict the agent's $(x, y)$ coordinates using **only** the Q-values as input.
    *   This forces the network to extract spatial information from the agent's policy representation.

3.  **Phase 3: Analysis of Emergent Representations**
    *   We analyze the hidden layer of the trained decoder network.
    *   Using standard neuroscience metrics (Spatial Information, Sparsity), we identify units that exhibit "place cell" properties—firing specifically when the agent is in a particular location.

## 🛠️ Installation

1.  **Clone the repository** or download the source code.
2.  **Install dependencies** using pip:

```bash
pip install -r requirements.txt
```

*Dependencies:* `numpy`, `matplotlib`, `torch`

## 💻 Usage

The project is designed to be run in two simple steps.

### Step 1: Train Models (`full_pipeline.py`)
This script handles the entire training workflow. It allows you to select an environment, trains the RL agent, and then trains the spatial decoder.

```bash
python full_pipeline.py
```

**Interactive Menu:**
You will be prompted to select an environment:
1.  **Open Field:** A standard square grid.
2.  **Four Rooms:** Four connected compartments (good for testing generalization).
3.  **T-Maze:** A classic neuroscience environment.
4.  **Random Barriers:** A grid with randomly placed obstacles.

*The script will save the trained models with filenames corresponding to the environment (e.g., `trained_agent_t_maze.pkl`).*

### Step 2: Analyze Results (`run_analysis.py`)
Once models are trained, this script performs a deep dive analysis of the hidden representations to find place cells.

```bash
python run_analysis.py
```

**Features:**
*   Calculates **Spatial Information (SI)** content for every hidden unit.
*   Computes **Sparsity** metrics.
*   Generates visualization plots (saved to `analysis_results/`).
*   Produces a detailed text report (`analysis_report.txt`).

## 📂 Project Structure

*   **Core Logic:**
    *   `q_learning.py`: Implementation of the Q-Learning agent.
    *   `decoder.py`: PyTorch implementation of the spatial decoder network.
    *   `environment.py`: Definitions for the GridWorld environments.
    *   `analysis.py`: Scientific metrics for place cell identification.
*   **Execution Scripts:**
    *   `full_pipeline.py`: Main training script (Phase 1 & 2).
    *   `run_analysis.py`: Main analysis script (Phase 3).
*   **Utilities:**
    *   `visualization.py`: Plotting functions for rate maps and training curves.

## 📊 Expected Results

If successful, the analysis should reveal that:
1.  **>30%** of hidden units exhibit spatially localized firing fields (Place Cells).
2.  These units have high **Spatial Information (>0.5 bits)**.
3.  The network effectively decodes position (**RMSE < 2.0 grid units**).

These results support the hypothesis that spatial maps can emerge as a byproduct of learning to navigate, without needing hard-coded spatial circuitry.
