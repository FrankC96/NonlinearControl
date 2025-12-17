import pygame
import numpy as np
import numpy.typing as npt

from typing import Tuple

FloatArray = npt.NDArray[np.float64]


def dynamics(x: FloatArray, u: FloatArray) -> FloatArray:
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


def generate_reference(
    t: int, circle_center: Tuple[int], circle_omega: float, A: int, B: int
) -> FloatArray:
    x_ref = circle_center[0] + A * np.cos(circle_omega * t)
    y_ref = circle_center[1] + B * np.sin(2 * circle_omega * t)

    theta_ref = np.arctan2(
        B * circle_omega * np.cos(circle_omega * t),
        -A * circle_omega * np.sin(circle_omega * t),
    )

    return np.array([x_ref, y_ref, theta_ref])


def wrap(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class HUD:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.s_width, self.s_height = self.screen.get_size()

        self.font = pygame.font.SysFont(None, 24)

    def precompute_reference(
        self, max_time: float, fps: int, reference_generator
    ) -> pygame.Surface:
        path_surface = pygame.Surface((self.s_width, self.s_height), pygame.SRCALPHA)
        for pt in reference_generator.orbit(max_time, fps):
            pygame.draw.circle(path_surface, (255, 255, 0), pt[:2], 1)
        return path_surface

    def blit_surface(self, surface: pygame.Surface):
        self.screen.blit(surface, (0, 0))

    def display_time(self, time: float):
        time_text = self.font.render(f"Time: [{time:.2f}s]", True, (255, 255, 255))
        self.screen.blit(source=time_text, dest=(int(self.s_width * 0.9), 10))

    def display_timestep(self, act_fps: float):
        dt_text = self.font.render(
            f"Timestep: [{act_fps:.2f}Hz]", True, (255, 255, 255)
        )
        self.screen.blit(source=dt_text, dest=(int(self.s_width * 0.9), 50))
