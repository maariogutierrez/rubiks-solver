"""
This module serves as the main entry point for the Rubik's Cube solver application.

It provides a command-line interface to either:
1. Train a neural network model to predict the distance to the solved state, or
2. Solve a scrambled Rubik's Cube using A* search with the trained model as a heuristic.

The application detects GPU availability and uses it if available for faster computation.
"""
from rubiks_engine import RubiksEngine
from nn import train, CubeValueNet
from solver import solve
import torch
import argparse

if __name__ == '__main__':
    # --- Command-Line Interface Setup ---
    # Parse arguments for training or solving mode
    parser = argparse.ArgumentParser(description='Train or solve Rubik\'s Cube')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-s', '--solve', type=str, help='Solve the cube with given moves (e.g., "U R U\' F")')
    parser.add_argument('-m', '--max-nodes', type=int, default=10000, help='Maximum nodes to explore in solve (default: 10000)')
    
    args = parser.parse_args()
    
    # --- Device Detection ---
    # Detect GPU availability and select the appropriate device for model computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.train:
        # --- Training Mode ---
        # Initialize a Rubik's Cube engine and neural network
        cube = RubiksEngine()
        net = CubeValueNet().to(device)
        
        # Train the network for 50000 epochs
        print("Starting model training...")
        train(net, cube, epochs=50000, device=device)
        
        # Save the trained model weights
        torch.save(net.state_dict(), 'new_model.pth')
        print("Model weights saved to new_model.pth")
    
    elif args.solve:
        # --- Solving Mode ---
        # 1. Load the trained neural network model
        model = CubeValueNet().to(device)
        weights = torch.load("model.pth", map_location=device)
        model.load_state_dict(weights)
        model.eval()  # Set model to evaluation mode (no gradient computation)
        print("Model weights loaded from model.pth")
        
        # 2. Initialize and scramble the cube with the provided moves
        engine = RubiksEngine()
        engine.execute(args.solve)
        print(f"Cube scrambled with: {args.solve}")
        
        # 3. Solve the cube using A* search with the neural network heuristic
        solution = solve(engine, model, device, max_nodes=args.max_nodes)
        if solution:
            print(f"Solution: {' -> '.join(solution)}")
        else:
            print("No solution found within the node limit.")
    
    else:
        # --- No Valid Mode Selected ---
        print("Please use -t to train or -s \"moves\" to solve the cube")
        parser.print_help()