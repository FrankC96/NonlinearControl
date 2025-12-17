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


class Simulation:
    def __init__(self, t: int, fps: int):
        dt_target = 1 / fps
        self.t = t
        self.fps = fps

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

        # FIXME: These all go to simulation setup / robot - controller setup
        x0 = np.array([200, 200, 0.0])
        xref = np.array([700.0, 700.0, 0.0])

        robot_lqr = Robot(dynamics, x0, dt_target, (255, 0, 0))
        robot_pid = Robot(dynamics, x0, dt_target, (255, 255, 255))

        v_pid = PIDController([5.0, 0.0, 0.0])
        w_pid = PIDController([5.0, 0.0, 0.0])

        ctr = LTVController(f=dynamics, type="ilqr")
        # --------------------------------------------------------------------------------

        # FIXME: Reference class?
        uref = np.zeros(2)
        circle_center = np.array([800, 400])
        A, B = 300, 300
        circle_omega = 0.4

        xref_hist = deque([], maxlen=2)

        # Precompute reference points
        ref_trajectory_path = []
        for k in range(self.fps * 100):
            t = k / self.fps
            x_ref, y_ref, theta_ref = generate_reference(
                t=t, circle_center=circle_center, circle_omega=circle_omega, A=A, B=B
            )
            ref_trajectory_path.append((x_ref, y_ref, theta_ref))

        # Create a transparent surface for the path
        path_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        for pt in ref_trajectory_path:
            pygame.draw.circle(path_surface, (255, 255, 0), pt[:2], 1)
        # --------------------------------------------------------------------------------

        for k in range(self.sim_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    k = self.sim_steps

            now = time.perf_counter()
            dt = now - (next_time - dt_target)

            screen.fill((0, 50, 50))
            hud.blit_surface(path_surface)

            # Actual time in simulation
            t = k * (dt_target)

            # compute [!]current target position
            traj_state = generate_reference(
                t=t, circle_center=circle_center, circle_omega=circle_omega, A=A, B=B
            )

            # FIXME: Also to Reference class
            # compute reference heading from direction of motion (use derivative)
            # small dt for reference derivative: 1/fps
            if len(xref_hist) > 0:
                prev = xref_hist[-1]
                dx_ref = traj_state[0] - prev[0]
                dy_ref = traj_state[1] - prev[1]
                theta_ref = wrap(np.arctan2(dy_ref, dx_ref))

            xref = np.array([traj_state[0], traj_state[1], theta_ref])
            xref_hist.append(xref.copy())

            dist_lqr = robot_lqr.distance_to_target(xref)
            dist_pid = robot_pid.distance_to_target(xref)

            x_lqr_robot, y_lqr_robot = robot_lqr.pose[:2]
            x_pid_robot, y_pid_robot = robot_pid.pose[:2]
            x_tar, y_tar = xref[:2]

            hud.display_time(k * dt_target)
            hud.display_timestep(1 / dt)

            theta_lqr_ref = wrap(np.arctan2(y_tar - y_lqr_robot, x_tar - x_lqr_robot))
            theta_pid_ref = wrap(np.arctan2(y_tar - y_pid_robot, x_tar - x_pid_robot))
            angle_err = wrap(theta_pid_ref - robot_pid.pose[2])

            v_ = v_pid.calculate(-dist_pid, 0.0, dt_target)
            w_ = w_pid.calculate(-angle_err, 0.0, dt_target)
            inp_pid = np.array([v_, w_])

            if len(xref_hist) > 1:
                dx = (xref_hist[1][0] - xref_hist[0][0]) / (dt_target)
                dy = (xref_hist[1][1] - xref_hist[0][1]) / (dt_target)
                dtheta = (xref_hist[1][2] - xref_hist[0][2]) / (dt_target)

                uref[0] = np.sqrt(dx**2 + dy**2)
                uref[1] = wrap(dtheta)

                inp_lqr = ctr.step(
                    x=robot_lqr.pose,
                    xref=xref,
                    u=uref,
                    dt=dt_target,
                    Qx=1e-1 * np.diag([1, 1, 1]),
                    Qz=1e-1 * np.diag([1, 1]),
                    R=1e0 * np.diag([1e0, 1e2]),
                )

            robot_lqr.move(u=inp_lqr)
            robot_pid.move(u=inp_pid)

            robot_lqr.draw(screen=screen)
            robot_pid.draw(screen=screen)

            pygame.draw.circle(
                screen, (255, 0, 0), circle_center.astype(int), 2, width=2
            )
            pygame.draw.circle(screen, (255, 255, 0), xref[:2].astype(int), 8)

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
