import os
import asyncio
import time
import pygame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import deque

from typing import List, Tuple, Optional

from nonlinearcontrol.control.pid_controller import PIDController
from nonlinearcontrol.control.mpc_controller import ModelPredictiveController
from nonlinearcontrol.control.ltv_controller import LTVController

from nonlinearcontrol.simulator.robot import Robot

from nonlinearcontrol.utils import *


class ReferenceGenerator:
    def __init__(
        self, type: str, circle_center: Tuple[int], circle_omega: float, A: int, B: int
    ):
        self.type = type
        self.circle_center = circle_center
        self.circle_omega = circle_omega
        self.A = A
        self.B = B

    def orbit(self, t: float, fps: int) -> List[FloatArray]:
        ref_trajectory_path = []
        for k in range(fps * 100):
            t = k / fps
            x_ref, y_ref, theta_ref = self.step(
                t=t,
            )
            ref_trajectory_path.append((x_ref, y_ref, theta_ref))

        return ref_trajectory_path

    def step(self, t: float) -> FloatArray:
        if self.type == "circle":
            w1 = self.circle_omega * t
            w2 = w1
        elif self.type == "figure_8":
            w1 = self.circle_omega * t
            w2 = 2 * w1

        x_ref = self.circle_center[0] + self.A * np.cos(w1)
        y_ref = self.circle_center[1] + self.B * np.sin(w2)

        theta_ref = wrap(
            np.arctan2(
                self.B * self.circle_omega * np.cos(w1),
                -self.A * self.circle_omega * np.sin(w2),
            )
        )
        return np.array([x_ref, y_ref, theta_ref])


class Simulation:
    def __init__(self, t: int, fps: int, reference: ReferenceGenerator):
        dt_target = 1 / fps
        self.t = t
        self.fps = fps
        self.ref_gen = reference

        self.sim_steps = int(self.t / (dt_target))

        self.robots: List[Optional[Robot]] = []
        self.controllers: List[Optional[object]] = []

    def register_robot(self, robot: Robot) -> Robot:
        self.robots.append(robot)

    def register_controller(self, controller: object) -> None:
        self.controllers.append(controller)

    async def run_sim(self):
        dt_target = 1.0 / self.fps
        next_time = time.perf_counter()

        pygame.init()

        # Screen settings
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen_width, screen_height = screen.get_size()

        hud = HUD(screen=screen)
        reference_surface = hud.precompute_reference(self.t, self.fps, self.ref_gen)

        xref_hist = deque([], maxlen=2)

        # --------------------------------------------------------------------------------

        for k in range(self.sim_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            now = time.perf_counter()
            dt = now - (next_time - dt_target)

            screen.fill((0, 50, 50))
            hud.blit_surface(reference_surface)

            # Actual time in simulation
            t = k * (dt_target)

            # compute [!]current target position
            traj_state = self.ref_gen.step(t)

            # FIXME: Also to Reference class
            # compute reference heading from direction of motion (use derivative)
            # small dt for reference derivative: 1/fps
            # if len(xref_hist) > 0:
            #     prev = xref_hist[-1]
            #     dx_ref = traj_state[0] - prev[0]
            #     dy_ref = traj_state[1] - prev[1]
            #     theta_ref = wrap(np.arctan2(dy_ref, dx_ref))

            # xref = np.array([traj_state[0], traj_state[1], theta_ref])
            # xref_hist.append(xref.copy())

            # dist_lqr = robot_lqr.distance_to_target(xref)
            # dist_pid = robot_pid.distance_to_target(xref)

            # x_lqr_robot, y_lqr_robot = robot_lqr.pose[:2]
            # x_pid_robot, y_pid_robot = robot_pid.pose[:2]
            # x_tar, y_tar = xref[:2]

            hud.display_time(k * dt_target)
            hud.display_timestep(1 / dt)

            # theta_lqr_ref = wrap(np.arctan2(y_tar - y_lqr_robot, x_tar - x_lqr_robot))
            # theta_pid_ref = wrap(np.arctan2(y_tar - y_pid_robot, x_tar - x_pid_robot))
            # angle_err = wrap(theta_pid_ref - robot_pid.pose[2])

            # v_ = v_pid.calculate(-dist_pid, 0.0, dt_target)
            # w_ = w_pid.calculate(-angle_err, 0.0, dt_target)
            # inp_pid = np.array([v_, w_])

            # if len(xref_hist) > 1:
            #     dx = (xref_hist[1][0] - xref_hist[0][0]) / (dt_target)
            #     dy = (xref_hist[1][1] - xref_hist[0][1]) / (dt_target)
            #     dtheta = (xref_hist[1][2] - xref_hist[0][2]) / (dt_target)

            #     uref[0] = np.sqrt(dx**2 + dy**2)
            #     uref[1] = wrap(dtheta)

            #     inp_lqr = ctr.step(
            #         x=robot_lqr.pose,
            #         xref=xref,
            #         u=uref,
            #         dt=dt_target,
            #         Qx=1e-1 * np.diag([1, 1, 1]),
            #         Qz=1e-1 * np.diag([1, 1]),
            #         R=1e0 * np.diag([1e0, 1e2]),
            #     )

            # robot_lqr.move(u=inp_lqr)
            # robot_pid.move(u=inp_pid)

            for robot in self.robots:
                robot.draw(screen=screen, x_curr=robot.pose, xref=traj_state[:2])

            # pygame.draw.circle(
            #     screen, (255, 0, 0), self.ref_gen.circle_center.astype(int), 2, width=2
            # )
            pygame.draw.circle(screen, (255, 255, 0), traj_state[:2].astype(int), 8)

            pygame.display.flip()

            next_time += dt_target
            delay = next_time - time.perf_counter()

            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # overrun loop running slower than target
                print("[!] Loop overrun")
                next_time = time.perf_counter()

            k += 1

        pygame.quit()
