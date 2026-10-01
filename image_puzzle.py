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
class ImageProcessor:
    @staticmethod
    def load(path: str) -> np.ndarray:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (OSError, cv2.error) as exc:
            raise ValueError(f"Could not read the file: {exc}") from exc
        if image is None:
            raise ValueError("That file is not a readable image (use JPG, PNG or BMP).")
        return image

    @staticmethod
    def prepare(image: np.ndarray, grid_size: int, target: int) -> np.ndarray:
        side = (target // grid_size) * grid_size
        h, w = image.shape[:2]
        edge = min(h, w)                                  
        top, left = (h - edge) // 2, (w - edge) // 2
        square = image[top:top + edge, left:left + edge]  
        interp = cv2.INTER_AREA if edge > side else cv2.INTER_CUBIC
        return cv2.resize(square, (side, side), interpolation=interp)

    @staticmethod
    def split(image: np.ndarray, grid_size: int) -> List[Tile]:
        step = image.shape[0] // grid_size
        tiles = []
        for r in range(grid_size):
            for c in range(grid_size):
                block = image[r * step:(r + 1) * step, c * step:(c + 1) * step]
                tiles.append(Tile(r * grid_size + c, np.ascontiguousarray(block)))
        return tiles


class Scrambler:
    TRANSFORM_COUNT = {3: 6, 4: 12, 5: 20}

    def __init__(self, rng: Optional[random.Random] = None) -> None:
       self._rng = rng or random.Random()

    def _make_swap(self, free: List[int]) -> Transformation:
       return SwapTransformation(free.pop(), free.pop())

    def _make_rotate(self, free: List[int]) -> Transformation:
       return RotateTransformation(free.pop(), self._rng.choice((1, 2, 3)))

    def _make_flip(self, free: List[int]) -> Transformation:
       return FlipTransformation(free.pop(), self._rng.choice((True, False)))

    def generate(self, grid_size: int) -> List[Transformation]:
        cells = grid_size * grid_size
        count = self.TRANSFORM_COUNT[grid_size]
        max_swaps = cells - count                      
        chosen = [self._make_swap, self._make_rotate, self._make_flip]   
        swaps = 1
        while len(chosen) < count:
            maker = self._rng.choice((self._make_swap, self._make_rotate, self._make_flip))
            if maker == self._make_swap:
                if swaps >= max_swaps:
                    continue
                swaps += 1
            chosen.append(maker)
        self._rng.shuffle(chosen)
        free = self._rng.sample(range(cells), cells)    
        return [make(free) for make in chosen]

    def scramble(self, board: PuzzleBoard) -> List[Transformation]:
        while True:
            board.solve()
            applied = self.generate(board.grid_size)
            for transformation in applied:
                board.apply(transformation)
            if not board.is_solved():
                return applied


# ==========================================================================
# Game logic (no Tkinter in here, so it can be tested on its own)
# ==========================================================================
class GameTimer:
    def __init__(self, limit_seconds: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit_seconds
        self._clock = clock
        self._start = clock()
        self._stopped_at: Optional[float] = None

    @property
    def limit(self) -> Optional[int]:
        return self._limit

    @property
    def elapsed(self) -> float:
        end = self._stopped_at if self._stopped_at is not None else self._clock()
        return end - self._start

    @property
    def remaining(self) -> Optional[float]:
        if self._limit is None:
            return None
        return max(0.0, self._limit - self.elapsed)

    @property
    def running(self) -> bool:
       return self._stopped_at is None

    def is_expired(self) -> bool:
       return self._limit is not None and self.elapsed >= self._limit

    def stop(self) -> None:
        if self._stopped_at is None:
            self._stopped_at = self._clock()

    @staticmethod
    def format(seconds: float) -> str:
        whole = int(seconds)
        return f"{whole // 60}:{whole % 60:02d}"


class PuzzleGame:
    MAX_HINTS = 3

    def __init__(self, image: np.ndarray, grid_size: int, time_limit: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._timer = GameTimer(time_limit, clock)
        self._timed_out = False
        self._original = image
        self._board = PuzzleBoard(ImageProcessor.split(image, grid_size), grid_size)
        self._transformations = Scrambler().scramble(self._board)
        self._moves = 0
        self._hints_used = 0
        self._hint: Optional[Tuple[int, int]] = None   
        self._selected: Optional[int] = None
        self._finished = False
        self._auto_solved = False

    #  read-only state 
    def grid_size(self) -> int:
        return self._board.grid_size

    @property
    def original_image(self) -> np.ndarray:
       return self._original

    @property
    def transformed_image(self) -> np.ndarray:
        return self._board.assemble()

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def incorrect_count(self) -> int:
        return len(self._board.incorrect_positions())

    @property
    def correct_positions(self) -> List[int]:
        return self._board.correct_positions()

    @property
    def selected(self) -> Optional[int]:
        return self._selected

    @property
    def hint(self) -> Optional[Tuple[int, int]]:
        return self._hint

    @property
    def hints_left(self) -> int:
        return self.MAX_HINTS - self._hints_used

    @property
    def hints_used(self) -> int:
        return self._hints_used

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def locked(self) -> bool:
        return self._finished or self._timed_out

    @property
    def time_limit(self) -> Optional[int]:
        return self._timer.limit

    @property
    def elapsed(self) -> float:
        return self._timer.elapsed

    @property
    def time_left(self) -> Optional[float]:
        return self._timer.remaining

    @property
    def auto_solved(self) -> bool:
        return self._auto_solved

    # -- player actions ---------------------------------------------------
    def _play(self, transformation: Transformation) -> None:
        self._board.apply(transformation)
        self._moves += 1
        self._hint = None                       
        if self._board.is_solved():
            self._finished = True
            self._selected = None
            self._timer.stop()

    def tick(self) -> None:
        if not self.locked and self._timer.is_expired():
            self._timed_out = True
            self._timer.stop()
            self._selected = None
            self._hint = None

    def click(self, position: int) -> None:
        self.tick()
        if self.locked:
            return
        if self._selected is None:
            self._selected = position
        elif self._selected == position:
            self._selected = None
        else:
            first, self._selected = self._selected, None
            self._play(SwapTransformation(first, position))

    def rotate(self, position: int) -> None:
        self.tick()
        if not self.locked:
            self._play(RotateTransformation(position, 1))

    def flip(self, position: int) -> None:
        self.tick()
        if not self.locked:
            self._play(FlipTransformation(position, horizontal=True))

    def use_hint(self) -> bool:
        self.tick()
        if self.locked or self._hint is not None or self.hints_left <= 0:
            return False
        position = random.choice(self._board.incorrect_positions())
        self._hint = (position, self._board.tile_at(position).home)
        self._hints_used += 1
        return True

    def solve(self) -> bool:
        self.tick()
        if self.locked:
            return False
        self._board.solve()
        self._moves = 0
        self._hint = None
        self._selected = None
        self._finished = True
        self._auto_solved = True
        self._timer.stop()
        return True

    def __str__(self) -> str:
        return (f"PuzzleGame {self.grid_size}x{self.grid_size}: {self._moves} moves, "
                f"{self.incorrect_count} tiles incorrect, {self.hints_left} hints left")


# ==========================================================================
# View layer (Tkinter)
# ==========================================================================
