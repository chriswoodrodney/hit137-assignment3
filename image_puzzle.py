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


