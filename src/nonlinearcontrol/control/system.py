import numpy as np
import numpy.typing as npt
import sympy as sp

from nonlinearcontrol.control.controller import *

from typing import Callable, List, Optional, Dict

SymbolicMatrix = sp.Matrix
FloatArray = npt.NDArray[np.float64]


class System:
    def __init__(self, dt: float):
        self.dt = dt

        self.state_vector: List[Optional[sp.Symbol]] = []
        self.input_vector: List[Optional[sp.Symbol]] = []

        self.rhs: List[Optional[sp.Symbol]] = []

        self.f_num: List[Optional[sp.Matrix]] = []

        self.controller_mapping: Dict[Optional[str], Optional[Controller]] = {}

    def register_system(self):
        self.nx = len(self.state_vector)
        self.nu = len(self.input_vector)

        self.rhs = sp.Matrix(self.rhs)

        self.f_num = sp.lambdify(
            (self.state_vector, self.input_vector),
            self.rhs,
            "numpy",
        )

    def add_system_state(self, x_var: str) -> sp.Symbol:
        x = sp.Symbol(x_var)
        self.state_vector.append(x)
        return x

    def add_system_input(self, u_var: str) -> sp.Symbol:
        u = sp.Symbol(u_var)
        self.input_vector.append(u)
        return u

    def add_system_rhs(self, eq: sp.Symbol) -> None:
        self.rhs.append(eq)

    def integrate(
        self, f: Callable, x: FloatArray, u: FloatArray, type: str = "RK4STEP"
    ):
        k1 = f(x, u).ravel()
        k2 = f(x + k1 * self.dt / 2, u).ravel()
        k3 = f(x + k2 * self.dt / 2, u).ravel()
        k4 = f(x + k3 * self.dt, u).ravel()

        return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def step(self, x: FloatArray, xref: FloatArray) -> FloatArray:
        u = np.zeros(self.nu)

        for idx, ctr in enumerate(self.controller_mapping.values()):
            error = ctr.error_func(x, xref)
            u[idx] = ctr.compute_control(error)

        return self.integrate(self.f_num, x, u)

    def register_controller(self, u: sp.Symbol, controller: Controller) -> None:
        self.controller_mapping[str(u)] = controller


def wrap(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def distance_to_target(
    robot_position: List[float], target_position: List[float]
) -> float:
    x, y, theta = robot_position
    x_tar, y_tar, theta_tar = target_position

    return np.sqrt((x_tar - x) ** 2 + (y_tar - y) ** 2)


def target_heading(robot_position: List[float], target_position: List[float]) -> float:
    x, y, theta = robot_position
    x_tar, y_tar, theta_tar = target_position

    return wrap(np.arctan2(y - y_tar, x - x_tar))


if __name__ == "__main__":
    DT = 1e-3
    # Initialize an empty system
    sys = System(dt=DT)

    # System variables
    x = sys.add_system_state("x")
    y = sys.add_system_state("y")
    theta = sys.add_system_state("theta")

    # Input variables
    v = sys.add_system_input("v")
    w = sys.add_system_input("w")

    # RHS equations
    sys.add_system_rhs(v * sp.cos(theta))
    sys.add_system_rhs(v * sp.sin(theta))
    sys.add_system_rhs(w)

    # Post initialization needed
    sys.register_system()

    # Create controllers
    v_pid = OutputFeedbackController(
        "PID", gains=[5.0, 1.0, 1.0], f_error=distance_to_target, dt=DT
    )
    w_pid = OutputFeedbackController(
        "PID", gains=[5.0, 1.0, 1.0], f_error=target_heading, dt=DT
    )

    # Assign controller to each input
    sys.register_controller(v, v_pid)
    sys.register_controller(w, w_pid)

    x0 = np.array([0.0, 0.0, 0.0])
    xtarget = np.array([100.0, 100.0, 0.0])

    x_next = sys.step(x0, xtarget)
    # print(x_next)
