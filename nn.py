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

def train_step(network, engine, optimizer, batch_size=128):
    """
    Performs a single training step for the network.

    This involves generating a batch of scrambled cube states, predicting their
    values, calculating the loss against the true scramble depth, and updating
    the network weights.

    Args:
        network (CubeValueNet): The network to train.
        engine (rubiks_engine.RubiksEngine): The cube engine for generating states.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        batch_size (int): The number of scrambled cubes to include in the batch.

    Returns:
        float: The calculated loss for the batch.
    """
    network.train() # Set to training mode
    optimizer.zero_grad()
    
    states_batch = []
    targets_batch = []
    
    # --- Data Generation ---
    # Generate a batch of scrambled cube states and their corresponding scramble depths.
    initial_state = engine.state.copy()
    for _ in range(batch_size):
        # Scramble the cube with a random number of moves (1 to 20).
        k = random.randint(1, 10)
        for _ in range(k):
            engine.execute(random.choice(['U', 'D', 'F', 'B', 'L', 'R']) + random.choice(['', "'"]))
        
        # Convert the state to a tensor and store it with its scramble depth (target).
        states_batch.append(state_to_tensor(engine.state))
        targets_batch.append([float(k)])
        
        # Reset the cube for the next scramble in the batch.
        engine.state = initial_state.copy()

    # --- Batch Conversion ---
    # Convert the lists of states and targets into PyTorch tensors.
    input_tensor = torch.stack(states_batch) # Shape: (batch_size, 324)
    target_tensor = torch.tensor(targets_batch, dtype=torch.float32)

    # --- Forward and Backward Pass ---
    # 1. Predict the value for the entire batch.
    predictions = network(input_tensor)

    # 2. Calculate the Mean Squared Error loss between predictions and actual depths.
    loss = nn.MSELoss()(predictions, target_tensor)
    
    # 3. Backpropagate the loss and update the network weights.
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(network, engine, epochs=1000, learning_rate=0.0001):
    """Train the CubeValueNet on Rubik's Cube states.
    
    Args:
        network (CubeValueNet): The instance of the network to be trained.
        engine (rubiks_engine.RubiksEngine): An instance of the Rubik's Cube engine
                                             used for generating scrambled states.
        epochs (int): The total number of training steps to perform. Defaults to 1000.
        learning_rate (float): The learning rate for the Adam optimizer. Defaults to 0.0001.
    
    Returns:
        list[float]: A list containing the loss value from each training epoch.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        loss = train_step(network, engine, optimizer)
        losses.append(loss)
        
        # Print progress every 100 epochs to monitor training.
        if (epoch + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss (last 100): {avg_loss:.6f}")
    
    print("Training finished.")
    return losses