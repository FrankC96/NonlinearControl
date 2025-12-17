from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from typing import Optional, Callable, List

FloatArray = npt.NDArray[np.float64]


class Controller(ABC):
    """A controller class interface"""

    def __init__(self, name: str):
        self.name = name
        self.error = 0.0

    @abstractmethod
    def reset(self) -> None:
        """Reset the controller's internal state"""
        pass

    @abstractmethod
    def compute_control(self, error: float) -> FloatArray:
        """Calculate the next control"""
        pass


class OutputFeedbackController(Controller):
    def __init__(
        self, controller_type: str, gains: List[float], f_error: Callable, dt: float
    ):
        super().__init__(controller_type + "-OutputFeedbackController")
        self.control_type = controller_type

        self.K = gains
        self.error_func = f_error
        self.dt = dt

        self.error: float = 0.0
        self.prev_error: float = 0.0
        self.int_error: float = 0.0

    def reset(self):
        self.error: float = 0.0
        self.prev_error: float = 0.0
        self.int_error: float = 0.0

    def compute_control(self, error: float) -> FloatArray:
        kp, ki, kd = self.K

        self.error = error

        if self.control_type == f"PID":
            self.int_error += self.error * self.dt
            self.int_error = max(-200, min(self.int_error, 200))
            der_error = (self.error - self.prev_error) / max(1e-5, self.dt)

            u_ = kp * self.error + ki * self.int_error + kd * der_error

        elif self.control_type == "SMC":
            der_error = (self.error - self.prev_error) / max(1e-5, self.dt)

            lamda, K, phi = self.K

            sigma = der_error + lamda * self.error
            u_ = K * np.tanh(sigma / phi)

        self.prev_error = self.error

        return u_


class StateFeedbackController(Controller):
    def __init__(self, controller_type: str):
        assert controller_type in [
            "LQR, iLQR"
        ], f"Currently supported controllers are: [LQR, iLQR]"

        super().__init__(controller_type + "-StateFeedbackController")
        pass


if __name__ == "__main__":
    ctr = OutputFeedbackController("PID")
    print(ctr.name)
