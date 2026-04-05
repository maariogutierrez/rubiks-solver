# Rubik's Cube Solver

A deep reinforcement learning agent that learns to solve the 3×3×3 Rubik's Cube. It trains a neural network to predict the minimum number of moves to a solved state, then uses that heuristic inside an A\* search to find solutions.

---

## How It Works

The solver combines two components:

1. **Value Network (`CubeValueNet`)** — A deep residual network that maps any cube state to an estimated move-distance to the solved state. The network is trained offline using a variant of the Bellman equation:

   ```
   V(s) = 1 + min_a V(s')    for all successor states s'
   ```

   Training uses **curriculum learning** (scramble depth increases gradually from 2 → 30 moves) and a **target network** (updated every 400 epochs) for stable bootstrapped targets.

2. **A\* Search (`solve`)** — At inference time, A\* searches over the cube's state space. The trained network serves as the heuristic `h(n)`, and `g(n)` is the actual move count from the start state. The search explores nodes by ascending `f(n) = g(n) + h(n)`.

### Architecture

| Component | Detail |
|---|---|
| Input | 54 stickers × 6 one-hot bits = **324-dimensional** vector |
| Input projection | Linear(324 → 1024) + ELU |
| Residual tower | 4 × ResidualBlock(1024) with LayerNorm + ELU + skip connections |
| Value head | Linear(1024 → 512) + ELU + Linear(512 → 1) |
| Total parameters | ~3.3M |
| Optimizer | Adam, lr=1e-4, weight decay=1e-5 |

### Cube Representation

The cube state is a `(6, 3, 3)` NumPy integer array. Faces are indexed as:

| Index | Face |
|---|---|
| 0 | Up (U) |
| 1 | Down (D) |
| 2 | Front (F) |
| 3 | Back (B) |
| 4 | Left (L) |
| 5 | Right (R) |

Each cell holds an integer 1–6 representing its sticker color.

---

## Project Structure

```
rubiks-solver/
├── main.py            # CLI entry point (train or solve)
├── rubiks_engine.py   # Cube state representation and move execution
├── nn.py              # Neural network architecture and training loop
├── solver.py          # A* search algorithm
├── model.pth          # Pre-trained model weights
└── training.log       # Sample training log (50k epochs on GPU)
```

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (CPU or CUDA)
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

## Usage

### Solve a Scrambled Cube

Load the pre-trained model and solve a cube scrambled with a sequence of moves:

```bash
python main.py --solve "R U R' U' R' F R2 U' R' U' R U R' F'"
```

The solver prints each move in the solution path and the number of nodes explored.

**Increase the node budget** for harder scrambles (default: 10,000):

```bash
python main.py --solve "R U R' U' F2 D L2" --max-nodes 50000
```

### Train a New Model

Train a new value network from scratch (saves checkpoints every 1,000 epochs):

```bash
python main.py --train
```

The trained weights are saved to `new_model.pth`. Training runs for 50,000 epochs with curriculum learning. A GPU is strongly recommended — training was originally run on CUDA and took several hours.

### CLI Reference

```
usage: main.py [-h] [-t] [-s SOLVE] [-m MAX_NODES]

options:
  -h, --help                  Show this help message and exit
  -t, --train                 Train a new model
  -s SOLVE, --solve SOLVE     Solve cube scrambled with given move sequence
  -m MAX_NODES, --max-nodes   Max nodes for A* search (default: 10000)
```

---

## Move Notation

Moves follow standard [Rubik's Cube notation](https://ruwix.com/the-rubiks-cube/notation/):

| Notation | Meaning |
|---|---|
| `U` | Up face, 90° clockwise |
| `U'` | Up face, 90° counter-clockwise |
| `U2` | Up face, 180° |
| `D`, `F`, `B`, `L`, `R` | Respective faces, same suffix rules |

---

## Training Details

The training log (`training.log`) records a run of ~11,000 epochs on a GPU. Key observations:

- Epochs 1–400: Scramble depth 2, loss converges near 0 quickly.
- Epochs 400–4,000: Depth ramps up 2→10, loss stays in the 0.05–0.6 range as the network adapts.
- Epochs 4,000–11,000: Depth reaches 26–29 moves; loss stabilizes ~0.15–0.20.

Full training (50,000 epochs, depth up to 30) is needed for the best-quality heuristic.

---

## Limitations

- **Optimality**: A\* with a learned (inadmissible) heuristic finds *a* solution, not necessarily the shortest one.
- **Node budget**: Very deeply scrambled cubes may require a large `--max-nodes` value to solve.
- **Training time**: Full training requires a GPU and several hours to reach good heuristic quality.

---

## License

This project is provided as-is for educational purposes.
