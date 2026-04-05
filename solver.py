"""
This module implements the A* search algorithm for solving a Rubik's Cube.

The `solve` function uses A* search combined with a neural network heuristic
to find an efficient solution path from any scrambled cube state to the solved state.
The heuristic function predicts the minimum number of moves required for any given
cube configuration, which guides the search toward the goal.
"""
import torch
import heapq
import numpy as np
from nn import state_to_tensor

def solve(engine, model, device, max_nodes=10000):
    """
    Solves a Rubik's Cube using A* search with a neural network heuristic.

    This function performs an A* search from the current cube state to the solved state.
    It explores nodes in order of their f-score (f = g + h, where g is the cost from
    the start and h is the heuristic estimate to the goal). The heuristic is provided
    by a trained neural network model. The search terminates when the solved state is
    found or the maximum node exploration limit is reached.

    Args:
        engine (RubiksEngine): The Rubik's Cube engine instance with the current state.
        model (CubeValueNet): A trained neural network model that predicts the distance
                              to the solved state from any given cube configuration.
        device (torch.device): The device to run the model on (CPU or GPU).
        max_nodes (int): The maximum number of nodes to explore before giving up.
                         Defaults to 10000.

    Returns:
        list: A list of move strings representing the solution path, or None if no
              solution is found within the node exploration limit.
              Each move is formatted as 'U', 'U\'', 'D', 'D\'', etc.
    """
    # 1. Define possible moves
    moves = ["U", "U'", "D", "D'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]
    
    # 2. Initialize the search frontier and visited set
    # The frontier is a priority queue storing tuples: (f_score, g_score, state_bytes, path)
    # We use state.tobytes() as a hashable key for the visited dictionary to track explored states.
    start_state_bytes = engine.state.tobytes()
    
    # Calculate the initial heuristic value using the neural network model
    with torch.no_grad():
        h_start = model(state_to_tensor(engine.state).unsqueeze(0).to(device)).item()
    
    # Initialize frontier with the start state
    frontier = [(h_start, 0, start_state_bytes, [])]
    visited = {start_state_bytes: 0}  # Maps state_bytes -> best_g_score found for that state
    
    # --- Main Search Loop ---
    nodes_explored = 0
    while frontier:
        # Pop the node with the lowest f-score from the frontier
        _, g, current_bytes, path = heapq.heappop(frontier)
        nodes_explored += 1
        
        # Reconstruct the current cube state from its byte representation
        current_state = np.frombuffer(current_bytes, dtype=int).reshape(6, 3, 3)
        engine.state = current_state.copy()
        
        # Check if the current state is the solved state
        if np.array_equal(engine.state, engine.solved_state):
            print(f"SOLVED in {len(path)} moves! Nodes explored: {nodes_explored}")
            return path

        # Stop if we've exceeded the maximum number of nodes to explore
        if nodes_explored >= max_nodes:
            break

        # 3. Generate and evaluate all neighbor states
        for move in moves:
            # Move pruning: Avoid immediately reversing the last move (e.g., U followed by U')
            # This optimization reduces redundant exploration.
            if path and move[0] == path[-1][0] and len(move) != len(path[-1]):
                continue
                
            # Apply the move and get the resulting neighbor state
            engine.state = current_state.copy()
            engine.execute(move)
            neighbor_bytes = engine.state.tobytes()
            new_g = g + 1
            
            # Update visited set if this state is new or we found a shorter path to it
            if neighbor_bytes not in visited or new_g < visited[neighbor_bytes]:
                visited[neighbor_bytes] = new_g
                
                # Compute the heuristic value using the neural network model
                # This estimates the minimum number of moves from this state to the goal
                input_tensor = state_to_tensor(engine.state).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = model(input_tensor).item()
                
                # Calculate f-score: f(n) = g(n) + h(n)
                # The f-score combines the actual cost from the start with the estimated cost to the goal
                new_f = new_g + h 
                heapq.heappush(frontier, (new_f, new_g, neighbor_bytes, path + [move]))

    # If we exit the loop without finding a solution or exceeding max_nodes
    print(f"Failed to solve within {max_nodes} nodes.")
    return None