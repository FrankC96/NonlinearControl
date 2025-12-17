import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from casadi import *

from typing import Tuple, Callable
from dataclasses import dataclass

from nonlinearcontrol.control.mpc_controller import ModelPredictiveController


def f(x, u, dt):
    return x + dt * np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


@dataclass
class SimulationParameters:
    dynamics: Callable
    T: int = 100
    N: int = 10
    dt: float = 0.01
    S: int = int(T / dt)
    Q: Tuple[float] = (1000.0, 1000.0, 1000.0)
    R: Tuple[float] = (1.0, 100.0)
    nx: int = len(Q)
    nu: int = len(R)
    controller: ModelPredictiveController = ModelPredictiveController(
        N, np.diag(Q), np.diag(R), dt
    )


class Simulator:
    def __init__(
        self, x0: Tuple[float], xref: Tuple[float], params: SimulationParameters, root
    ):
        self.x0 = x0
        self.xr = xref

        self.params = params
        self.controller = self.params.controller

        x1, x1_ref = self.controller.add_state_variable("x1")
        x2, x2_ref = self.controller.add_state_variable("x2")
        x3, x3_ref = self.controller.add_state_variable("x3")
        u1 = self.controller.add_input_variable("u1")
        u2 = self.controller.add_input_variable("u2")

        self.controller.add_rhs(u1 * cos(x3))
        self.controller.add_rhs(u1 * sin(x3))
        self.controller.add_rhs(u2)

        x = vertcat(x1, x2, x3)
        x_ref = vertcat(x1_ref, x2_ref, x3_ref)
        u = vertcat(u1, u2)

        self.controller.add_cost_func(
            sum(
                (x - x_ref).T @ self.controller.Q @ (x - x_ref)
                + u.T @ self.controller.R @ u
            )
        )

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Initial plot
        x = np.linspace(0, 2 * np.pi, 100)
        (self.line,) = self.ax.plot(x, np.sin(x))

        # Embed the Matplotlib figure into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button to update the plot
        button = ttk.Button(root, text="Update Plot", command=self.simulate)
        button.pack(pady=10)

    def simulate(self):
        f = self.params.dynamics
        controller = self.params.controller

        X, U = np.empty([self.params.S, self.params.nx]), np.empty(
            [self.params.S, self.params.nu]
        )
        X[0] = self.x0
        for k in range(self.params.T):
            print(k)
            x_opt, u_opt = controller.make_step(X[k].tolist(), self.xr)
            U[k] = u_opt[0, :]
            X[k + 1] = f(X[k], U[k], self.params.dt)


if __name__ == "__main__":
    ROOT = tk.Tk()
    PARAMS = SimulationParameters(dynamics=f)
    app = Simulator(x0=[0, 0, 0], xref=[10, 10, 10], params=PARAMS, root=ROOT)
    ROOT.mainloop()
