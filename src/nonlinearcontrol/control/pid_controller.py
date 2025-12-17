from typing import List


class PIDController:
    def __init__(self, K: List[float]):
        self.kp, self.ki, self.kd = K

        self.reset()

    def reset(self) -> None:
        self.y_prev: float = 0.0
        self.ref_prev: float = 0.0

        self.int_error: float = 0.0

    def calculate(self, y: float, ref: float, dt: float) -> float:
        error = ref - y

        self.y_prev = y
        self.ref_prev = ref
        prev_error = self.ref_prev - self.y_prev

        self.int_error += error * dt
        self.int_error = max(-200, min(self.int_error, 200))

        return (
            self.kp * error
            + self.ki * self.int_error
            + self.kd * (error - prev_error) / max(1e-5, dt)
        )
