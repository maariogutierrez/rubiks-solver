"""
This module defines the neural network architecture and training loop for a Rubik's Cube solver.

The core component is the `CubeValueNet`, a deep residual network designed to
predict the "value" of a given cube state. The value is defined as the minimum
number of moves required to solve the cube (God's Number for that state).

The network takes a one-hot representation of the cube's 54 sticker colors
as input and outputs a single scalar value representing the predicted distance
to the solved state.
"""
import torch
import torch.nn as nn
import random
import copy
import numpy as np

class ResidualBlock(nn.Module):
    """
    A standard residual block with two linear layers, layer normalization, and ELU activation.
    The block implements a skip connection, which helps in training deep networks by
    allowing gradients to flow more easily.

    Args:
        dim (int): The dimension of the input and output features.
    """
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Performs the forward pass through the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the residual connection.
        """
        # This is the "Skip Connection" (Residual)
        return self.elu(x + self.layers(x))

class CubeValueNet(nn.Module):
    """
    A deep neural network that predicts the distance to the solved state for a Rubik's Cube.

    The network architecture consists of:
    1. An input layer that projects the one-hot encoded cube state to a higher dimension.
    2. A tower of residual blocks to process the features.
    3. An output head that predicts a single scalar value (the distance).

    Args:
        num_res_blocks (int): The number of residual blocks in the tower. Defaults to 4.
    """
    def __init__(self, num_res_blocks=4):
        super().__init__()
        # 1. Input Layer: Project One-Hot (324) to the hidden dimension (1024)
        self.input_layer = nn.Sequential(
            nn.Linear(324, 1024),
            nn.ELU()
        )
        
        # 2. Residual Tower: A series of residual blocks to learn complex features.
        self.res_tower = nn.Sequential(
            *[ResidualBlock(1024) for _ in range(num_res_blocks)]
        )
        
        # 3. Output Head: Predicts the distance (value) from the learned features.
        self.value_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor representing the cube state(s).
                               Shape should be (batch_size, 324) or (324,).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the predicted
                          distance to the solved state for each cube in the batch.
        """
        # Ensure input is flattened if coming from a batch
        x = x.view(x.size(0), -1) 
        x = self.input_layer(x)
        x = self.res_tower(x)
        return self.value_head(x)

def state_to_tensor(state):
    """
    Converts a 6x3x3 numpy array representing the cube state into a 324-element
    one-hot encoded PyTorch tensor.

    Each of the 54 stickers is represented by a 6-element one-hot vector,
    and these are concatenated to form a single flat tensor.

    Args:
        state (numpy.ndarray): A 6x3x3 array with integer values from 1 to 6
                               representing the colors of the cube faces.

    Returns:
        torch.Tensor: A 1D tensor of length 324 (54 stickers * 6 colors).
    """
    # Convert 1-6 to 0-5 indices for one-hot encoding
    t = torch.tensor(state - 1, dtype=torch.long)
    # One-hot encode: resulting shape (6, 3, 3, 6)
    one_hot = torch.nn.functional.one_hot(t, num_classes=6).float()
    return one_hot.view(-1) # Flatten to 324

# --- Constants ---
# All 12 possible moves on a standard Rubik's Cube (6 faces × 2 directions each)
MOVES = ["U", "U'", "D", "D'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]

def get_all_neighbors(engine, state):
    """
    Generates all 12 possible next states from the current state.

    This function applies each of the 12 possible moves to the current cube state
    and returns the resulting neighbor states as a stacked tensor. The neighbors
    are converted to one-hot encoded tensors suitable for neural network input.

    Args:
        engine (RubiksEngine): The Rubik's Cube engine instance.
        state (np.ndarray): A 6x3x3 numpy array representing the current cube state.

    Returns:
        torch.Tensor: A tensor of shape (12, 324) containing the one-hot encoded
                      representations of all 12 neighbor states.
    """
    neighbors = []
    original_state = state.copy()
    
    # Apply each of the 12 possible moves and collect the resulting states
    for m in MOVES:
        engine.state = original_state.copy()
        engine.execute(m)
        neighbors.append(state_to_tensor(engine.state))
        
    engine.state = original_state  # Reset engine to original state
    return torch.stack(neighbors)  # Stack into (12, 324) tensor

def train_step(network, target_network, engine, optimizer, batch_size=128, device="cpu", max_k=20):
    """
    Performs a single training step for the network using the Bellman equation.

    This function implements the core training loop:
    1. Generates a batch of random cube states by scrambling solved cubes
    2. Computes target values using the Bellman equation: target = 1 + min(V(neighbors))
    3. Trains the network to predict these target values

    The process uses a target network (separate copy of the main network) to compute
    target values, which stabilizes training by reducing moving target issues.

    Args:
        network (CubeValueNet): The main network being trained.
        target_network (CubeValueNet): A copy of the network used to compute targets.
        engine (RubiksEngine): The Rubik's Cube engine for generating states.
        optimizer (torch.optim.Optimizer): The optimizer for updating network weights.
        batch_size (int): Number of training samples per step. Defaults to 128.
        device (torch.device or str): Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
        max_k (int): Maximum number of random moves to scramble each cube. Defaults to 20.

    Returns:
        float: The MSE loss for this training step.
    """
    # Set network to training mode
    network.train()
    optimizer.zero_grad()
    
    states_batch = []
    targets_batch = []
    
    # 1. Experience Generation: Create scrambled states and compute Bellman targets
    for _ in range(batch_size):
        # Scramble a random amount (1-max_k moves)
        k = random.randint(1, max_k)
        for _ in range(k):
            engine.execute(random.choice(MOVES))
        
        current_state = engine.state.copy()
        
        # Check if we accidentally reached the solved state
        if np.array_equal(current_state, engine.solved_state):
            # Solved state has value 0 (no moves needed)
            target = 0.0
        else:
            # 2. Calculate Bellman Target: 1 + min(V(neighbors))
            # Get all 12 possible next states from this position
            neighbor_tensors = get_all_neighbors(engine, current_state)
            neighbor_tensors = neighbor_tensors.to(device)
            
            with torch.no_grad():
                # Use TARGET_NETWORK (not the training network) to compute neighbor values
                # This stabilizes training by providing consistent targets
                neighbor_values = target_network(neighbor_tensors)  # Shape: (12, 1)
                best_neighbor_val = torch.min(neighbor_values)
                # Target: 1 move + minimum cost from neighbors, but at least 0
                target = 1.0 + max(0, float(best_neighbor_val))
        
        states_batch.append(state_to_tensor(current_state))
        targets_batch.append([target])
        
        # Reset for next sample
        engine.state = engine.solved_state.copy()

    # 3. Optimize: Update network weights by minimizing MSE loss
    input_tensor = torch.stack(states_batch).to(device)
    target_tensor = torch.tensor(targets_batch, dtype=torch.float32).to(device)
    
    # Compute predictions and loss
    predictions = network(input_tensor)
    loss = torch.nn.MSELoss()(predictions, target_tensor)
    
    # Backpropagate and update weights
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(network, engine, epochs=1000, learning_rate=0.0001, device="cpu"):
    """
    Trains the neural network to predict the minimum number of moves to solve a Rubik's Cube.

    This function implements the full training loop for the value network using the Bellman
    equation. The network is trained to predict the distance to the solved state for any
    given cube configuration. Training progresses by:
    1. Gradually increasing the maximum scramble depth (from 2 to 30 moves)
    2. Regularly updating the target network for stable learning
    3. Saving checkpoints every 1000 epochs

    Args:
        network (CubeValueNet): The neural network to train.
        engine (RubiksEngine): The Rubik's Cube engine for generating training states.
        epochs (int): Number of training epochs to run. Defaults to 1000.
        learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.0001.
        device (torch.device or str): Device to train on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None. Training results are printed to console and checkpoints are saved to disk.
    """
    # --- Setup Phase ---
    # Initialize target network with same weights as the main network
    # The target network is updated periodically and used to compute stable training targets
    target_network = copy.deepcopy(network).to(device)
    target_network.eval()  # Set to evaluation mode (no dropout, no batch norm updates)
    
    # Create optimizer with L2 regularization
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # --- Training Loop ---
    for epoch in range(epochs):
        # Curriculum learning: gradually increase scramble depth over training
        # Starts at 2 moves, increases by 1 every 400 epochs, capped at 30 moves
        max_k = min(30, 2 + (epoch // 400))
        
        # Perform one training step
        loss = train_step(network, target_network, engine, optimizer, device=device, max_k=max_k)
            
        # --- Logging Phase ---
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | Max Scramble: {max_k} | Loss: {loss:.4f}")

        # --- Checkpointing Phase ---
        if (epoch + 1) % 1000 == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")
        
        # --- Target Network Update ---
        # Update the target network every 400 epochs with the latest main network weights
        # This periodically stabilizes the training targets used in the Bellman equation
        if epoch % 400 == 0:
            target_network.load_state_dict(network.state_dict())
            print("--- Target Network Updated ---")