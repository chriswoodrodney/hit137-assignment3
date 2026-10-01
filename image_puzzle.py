from __future__ import annotations

import base64
import math
import random
import time
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

# ==========================================================================
# Configuration
# ==========================================================================
DEFAULT_GRID = 3
GRID_CHOICES = (3, 4, 5)

SELECT_COLOUR = "#ff9800"   
TICK_COLOUR = "#1faa3c"     
HINT_COLOUR = "#1e78ff"     


# ==========================================================================
# Model layer: tiles, orientation, transformations and the board
# ==========================================================================
class Orientation:
    def __init__(self, quarter_turns: int = 0, mirrored: bool = False) -> None:
       self._turns = quarter_turns % 4
       self._mirrored = bool(mirrored)

    #  queries 
    @property
    def turns(self) -> int:
        return self._turns

    @property
    def mirrored(self) -> bool:
       return self._mirrored

    def is_identity(self) -> bool:
        return self._turns == 0 and not self._mirrored

    #  mutators 
    def rotate_clockwise(self, quarter_turns: int = 1) -> None:
       self._turns = (self._turns + quarter_turns) % 4

    def flip_horizontal(self) -> None:
        self._turns = (-self._turns) % 4
        self._mirrored = not self._mirrored

    def flip_vertical(self) -> None:
        self.flip_horizontal()
        self.rotate_clockwise(2)

    def reset(self) -> None:
        self._turns = 0
        self._mirrored = False

    def __str__(self) -> str:
        return ("mirrored, " if self._mirrored else "") + f"rotated {90 * self._turns} deg clockwise"

    #  OpenCV -----------------------------------------------------------
    def apply_to(self, pixels: np.ndarray) -> np.ndarray:
        out = cv2.flip(pixels, 1) if self._mirrored else pixels
        for _ in range(self._turns):
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        return out


class Tile:
    def __init__(self, home_index: int, pixels: np.ndarray) -> None:
        self._home = home_index
        self._pixels = pixels
        self._orientation = Orientation()

    @property
    def home(self) -> int:
       return self._home

    def rotate(self, quarter_turns: int = 1) -> None:
        self._orientation.rotate_clockwise(quarter_turns)

    def flip(self, horizontal: bool = True) -> None:
        if horizontal:
            self._orientation.flip_horizontal()
        else:
            self._orientation.flip_vertical()

    def is_upright(self) -> bool:
        return self._orientation.is_identity()

    def is_mirrored(self) -> bool:
       return self._orientation.mirrored

    def render(self) -> np.ndarray:
        return self._orientation.apply_to(self._pixels)

    def is_correct_at(self, position: int) -> bool:
        return position == self._home and self._orientation.is_identity()

    def restore(self) -> None:
       self._orientation.reset()

    def __str__(self) -> str:
        return f"Tile(home={self._home}, {self._orientation})"


class Transformation(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def apply(self, board: "PuzzleBoard") -> None:
        """Apply this transformation to ``board`` (every subclass implements this)."""
        ...

    def describe(self) -> str:
         return self._name

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:  # handy when debugging
        return f"<{self.describe()}>"


class SwapTransformation(Transformation):
    def __init__(self, first: int, second: int) -> None:
        super().__init__("Swap")
        self._first, self._second = first, second

    def apply(self, board: "PuzzleBoard") -> None:
        board.swap(self._first, self._second)

    def describe(self) -> str:
        return f"{super().describe()}: positions {self._first} and {self._second}"


class RotateTransformation(Transformation):
    def __init__(self, position: int, quarter_turns: int = 1) -> None:
        super().__init__("Rotate")
        self._position, self._turns = position, quarter_turns

    def apply(self, board: "PuzzleBoard") -> None:
         board.rotate(self._position, self._turns)

    def describe(self) -> str:
       return f"{super().describe()}: position {self._position} by {90 * self._turns} degrees clockwise"


class FlipTransformation(Transformation):
    def __init__(self, position: int, horizontal: bool = True) -> None:
        super().__init__("Flip")
        self._position, self._horizontal = position, horizontal

    def apply(self, board: "PuzzleBoard") -> None:
        board.flip(self._position, self._horizontal)

    def describe(self) -> str:
        axis = "horizontally" if self._horizontal else "vertically"
        return f"{super().describe()}: position {self._position} {axis}"


class PuzzleBoard:
   def __init__(self, tiles: List[Tile], grid_size: int) -> None:
    if len(tiles) != grid_size * grid_size:
            raise ValueError("tile count does not match grid size")
    self._grid = grid_size
    self._cells: List[Tile] = list(tiles)

    @property
    def grid_size(self) -> int:
        return self._grid

    def tile_at(self, position: int) -> Tile:
       return self._cells[position]

    # basic operations 
    def swap(self, a: int, b: int) -> None:
       self._cells[a], self._cells[b] = self._cells[b], self._cells[a]

    def rotate(self, position: int, quarter_turns: int = 1) -> None:
        self._cells[position].rotate(quarter_turns)

    def flip(self, position: int, horizontal: bool = True) -> None:
       self._cells[position].flip(horizontal)

    def apply(self, transformation: Transformation) -> None:
        transformation.apply(self)          

    # state 
    def is_correct(self, position: int) -> bool:
        return self._cells[position].is_correct_at(position)

    def correct_positions(self) -> List[int]:
        return [p for p in range(len(self._cells)) if self.is_correct(p)]

    def incorrect_positions(self) -> List[int]:
       return [p for p in range(len(self._cells)) if not self.is_correct(p)]

    def is_solved(self) -> bool:
        return not self.incorrect_positions()

    def solve(self) -> None:
        self._cells.sort(key=lambda tile: tile.home)
        for tile in self._cells:
            tile.restore()

    def __str__(self) -> str:
        g = self._grid
        rows = []
        for r in range(g):
            cells = []
            for c in range(g):
                tile = self._cells[r * g + c]
                cells.append(f"{tile.home:>2}{'' if tile.is_upright() else '*'}".ljust(3))
            rows.append(" ".join(cells).rstrip())
        return "\n".join(rows)

    # OpenCV 
    def assemble(self) -> np.ndarray:
        g = self._grid
        rows = [cv2.hconcat([self._cells[r * g + c].render() for c in range(g)])
                for r in range(g)]
        return cv2.vconcat(rows)


# ==========================================================================
# Image processing (OpenCV) and scrambling
# ==========================================================================
