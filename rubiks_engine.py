"""
This module provides a `RubiksEngine` class to represent and manipulate a 3x3x3
Rubik's Cube. The cube's state is stored in a NumPy tensor, and the class
provides methods to perform face rotations and display the cube's state.

The cube faces are indexed and ordered as follows:
- 0: Up (U)
- 1: Down (D)
- 2: Front (F)
- 3: Back (B)
- 4: Left (L)
- 5: Right (R)

Each face is a 3x3 matrix, making the entire cube state a (6, 3, 3) NumPy array.
The values in the array represent the color of each sticker.
"""

import numpy as np

class RubiksEngine:
    """
    Represents and manipulates the state of a 3x3x3 Rubik's Cube.

    The cube is represented by a (6, 3, 3) NumPy array where each of the six
    3x3 matrices corresponds to a face. The engine is initialized to a solved
    state and can be modified by applying move sequences.

    Attributes:
        state (np.ndarray): A (6, 3, 3) integer array representing the cube's
                            current sticker configuration.
        U, D, F, B, L, R (int): Integer constants for face indices.
        face_map (dict): A mapping from face characters ('U', 'D', etc.) to
                         their corresponding integer indices.
        rings (dict): A pre-calculated mapping of the 12 adjacent sticker
                      coordinates for each face, used for rotations.
    """

    def __init__(self):
        """Initializes the Rubik's Cube engine to a solved state."""
        # --- Face Constants and Mappings ---
        self.U, self.D, self.F, self.B, self.L, self.R = 0, 1, 2, 3, 4, 5
        self.face_map = {'U': 0, 'D': 1, 'F': 2, 'B': 3, 'L': 4, 'R': 5}
        
        # --- State Initialization ---
        # Initialize a solved cube where each face `i` is filled with color `i+1`.
        self.state = np.zeros((6, 3, 3), dtype=int)
        for face in range(6):
            self.state[face] = np.full((3, 3), face + 1)
        
        # Store the solved state for reference/comparison
        self.solved_state = self.state.copy()
            
        # --- Adjacency Definitions (Rings) ---
        # Define the 12 adjacent sticker coordinates for each face in CLOCKWISE order.
        # This is the core data structure that defines how faces are connected.
        # Each tuple is in the format: (Face Index, Row, Column)
        self.rings = {
            'U': [
                (self.B,0,2), (self.B,0,1), (self.B,0,0), # Back face, top row
                (self.R,0,2), (self.R,0,1), (self.R,0,0), # Right face, top row
                (self.F,0,2), (self.F,0,1), (self.F,0,0), # Front face, top row
                (self.L,0,2), (self.L,0,1), (self.L,0,0)  # Left face, top row
            ],
            'D': [
                (self.F,2,0), (self.F,2,1), (self.F,2,2), # Front face, bottom row
                (self.R,2,0), (self.R,2,1), (self.R,2,2), # Right face, bottom row
                (self.B,2,0), (self.B,2,1), (self.B,2,2), # Back face, bottom row
                (self.L,2,0), (self.L,2,1), (self.L,2,2)  # Left face, bottom row
            ],
            'F': [
                (self.U,2,0), (self.U,2,1), (self.U,2,2), # Up face, bottom row
                (self.R,0,0), (self.R,1,0), (self.R,2,0), # Right face, left column
                (self.D,0,2), (self.D,0,1), (self.D,0,0), # Down face, top row (reversed)
                (self.L,2,2), (self.L,1,2), (self.L,0,2)  # Left face, right column (reversed)
            ],
            'B': [
                (self.U,0,2), (self.U,0,1), (self.U,0,0), # Up face, top row (reversed)
                (self.L,0,0), (self.L,1,0), (self.L,2,0), # Left face, left column
                (self.D,2,0), (self.D,2,1), (self.D,2,2), # Down face, bottom row
                (self.R,2,2), (self.R,1,2), (self.R,0,2)  # Right face, right column (reversed)
            ],
            'L': [
                (self.U,0,0), (self.U,1,0), (self.U,2,0), # Up face, left column
                (self.F,0,0), (self.F,1,0), (self.F,2,0), # Front face, left column
                (self.D,0,0), (self.D,1,0), (self.D,2,0), # Down face, left column
                (self.B,2,2), (self.B,1,2), (self.B,0,2)  # Back face, right column (reversed)
            ],
            'R': [
                (self.U,2,2), (self.U,1,2), (self.U,0,2), # Up face, right column (reversed)
                (self.B,0,0), (self.B,1,0), (self.B,2,0), # Back face, left column
                (self.D,2,2), (self.D,1,2), (self.D,0,2), # Down face, right column (reversed)
                (self.F,2,2), (self.F,1,2), (self.F,0,2)  # Front face, right column (reversed)
            ]
        }

    def execute(self, sequence):
        """
        Apply a sequence of face turns to the current cube state.

        The sequence is a string of space-separated moves.

        Args:
            sequence (str): A string of moves like "R U R' U'".
                Supported move notations:
                - `X`: Clockwise 90-degree turn of face X.
                - `X'`: Counter-clockwise 90-degree turn of face X.
                - `X2`: 180-degree turn of face X.
                Where X is one of U, D, F, B, L, R.

        Raises:
            ValueError: If a move in the sequence is malformed or unsupported.
            KeyError: If a face character is not a valid face identifier.
        """
        moves = sequence.split()
        for move in moves:
            face = move[0]
            if len(move) == 1:
                self._rotate(face, 1)    # 1 quarter turn clockwise
            elif move[1] == "'":
                self._rotate(face, -1)   # 1 quarter turn counter-clockwise
            elif move[1] == '2':
                self._rotate(face, 2)    # 2 quarter turns (180 degrees)
            else:
                raise ValueError(f"Invalid move: {move}")

    def _rotate(self, face_name, turns):
        """
        Rotates a single face and cycles its adjacent ring of stickers.

        This is the core logic for performing a move. It involves two steps:
        1. Rotating the stickers on the face itself.
        2. Permuting the 12 stickers on the adjacent faces (the "ring").

        Args:
            face_name (str): The character of the face to rotate ('U', 'D', etc.).
            turns (int): The number of quarter-turns to apply.
                         -  1: Clockwise
                         - -1: Counter-clockwise
                         -  2: 180-degree turn
        """
        face_idx = self.face_map[face_name]
        
        # 1. Rotate the stickers on the face itself.
        # np.rot90's `k` parameter is for counter-clockwise turns. We negate our
        # `turns` value to map our convention (1=CW) to numpy's (1=CCW).
        self.state[face_idx] = np.rot90(self.state[face_idx], k=-turns)

        # 2. Cycle the 12 adjacent stickers in the ring.
        ring = self.rings[face_name]
        
        # Get the current colors of the stickers in the ring.
        current_values = [self.state[f, r, c] for f, r, c in ring]
        
        # A 90-degree turn (1 turn) shifts the 12 stickers by 3 positions.
        shift = 3 * turns
        
        # Apply the shifted values back to the cube state.
        # The modulo operator (%) ensures the indices wrap around correctly.
        for i, (f, r, c) in enumerate(ring):
            self.state[f, r, c] = current_values[(i - shift) % 12]

    def display(self):
        """
        Prints a 2D "unfolded" representation of the cube's state to the console.

        This is useful for debugging and visualizing the cube in a terminal.
        The layout is as follows:
              U
            L F R B
              D
        """
        empty_row = " " * 8
        
        # Print Up face
        for row in self.state[self.U]:
            print(f"{empty_row}{row}")
            
        # Print Left, Front, Right, Back faces in a row
        for i in range(3):
            l_row = self.state[self.L][i]
            f_row = self.state[self.F][i]
            r_row = self.state[self.R][i]
            b_row = self.state[self.B][i]
            print(f"{l_row} {f_row} {r_row} {b_row}")
            
        # Print Down face
        for row in self.state[self.D]:
            print(f"{empty_row}{row}")
        print("-" * 40)