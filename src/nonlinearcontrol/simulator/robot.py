import pygame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Callable, List, Tuple

from pygame import draw


type FloatArray = npt.NDArray[np.float64]


class Robot:
    def __init__(
        self,
        name: str,
        sys: "System",
        x: FloatArray,
        dt: float,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.name = name
        self.sys = sys
        self.pose = x.copy()
        self.input = None
        self.dt = dt
        self.color = color

        self.font = pygame.font.SysFont(None, 24)

        # Store the previous poses and inputs applied
        self.hist_pose = []
        self.hist_inp = []

    def draw(self, screen, x: FloatArray, xref: FloatArray) -> None:
        x_next = self.sys.step(x, xref)
        x, y, theta = x_next

        self.hist_pose.append(x_next)
        x_end, y_end = self.pose[:2] + 30.0 * np.array([np.cos(theta), np.sin(theta)])

        draw.circle(screen, self.color, (x, y), 30.0)
        draw.line(screen, (0, 0, 0), (x, y), np.array([x_end, y_end]))

        for idx, point in enumerate(self.hist_pose):
            x_prev, y_prev, _ = point
            if idx % 2 == 0 and idx < len(x_prev) - 2:
                draw.circle(screen, self.color, (x_prev, y_prev), 2.5)
            else:
                continue
