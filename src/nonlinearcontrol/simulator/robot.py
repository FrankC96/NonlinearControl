import pygame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Callable, List, Tuple

from pygame import draw


FloatArray = npt.NDArray[np.float64]


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

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)

        # Store the previous poses and inputs applied
        self.hist_pose = []
        self.hist_inp = []

    def draw(self, screen, x_curr: FloatArray, xref: FloatArray) -> None:
        pose = self.sys.step(x=x_curr, xref=xref)
        x, y, theta = pose

        self.hist_pose.append(self.pose.copy())
        x_end, y_end = self.pose[:2] + 30.0 * np.array([np.cos(theta), np.sin(theta)])

        draw.circle(screen, self.color, (x, y), 30.0)
        draw.line(screen, (0, 0, 0), (x, y), (x_end, y_end))

        self.pose = pose

        for idx, point in enumerate(self.hist_pose):
            x_prev, y_prev, _ = point
            if idx % 2 == 0 and idx < len(self.hist_pose) - 2:
                draw.circle(screen, self.color, (x_prev, y_prev), 2.5)
            else:
                continue
