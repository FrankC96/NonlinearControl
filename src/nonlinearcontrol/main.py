import asyncio

from nonlinearcontrol.control.system import *
from nonlinearcontrol.control.controller import *
from nonlinearcontrol.simulator.robot import *

from nonlinearcontrol.simulator.sim import *


if __name__ == "__main__":
    TIME = 100
    FPS = 30
    DT = 1 / FPS

    REFERENCE_GENERATOR = ReferenceGenerator(
        type="figure_8", circle_center=(800, 400), circle_omega=0.3, A=400, B=300
    )

    SYSTEM = System(DT)
    # System variables
    x = SYSTEM.add_system_state("x")
    y = SYSTEM.add_system_state("y")
    theta = SYSTEM.add_system_state("theta")

    # Input variables
    v = SYSTEM.add_system_input("v")
    w = SYSTEM.add_system_input("w")

    # RHS equations
    SYSTEM.add_system_rhs(v * sp.cos(theta))
    SYSTEM.add_system_rhs(v * sp.sin(theta))
    SYSTEM.add_system_rhs(w)

    # Post initialization needed
    SYSTEM.register_system()

    # Create controllers
    v_pid = OutputFeedbackController(
        "SMC", gains=[200.0, 160.0, 20.0], f_error=distance_to_target, dt=DT
    )
    # v_pid = OutputFeedbackController(
    #     "PID", gains=[5.0, 0.0, 0.0], f_error=distance_to_target, dt=DT
    # )
    w_pid = OutputFeedbackController(
        "SMC", gains=[5.0, 8.0, 20.0], f_error=target_heading, dt=DT
    )

    # Assign controller to each input
    SYSTEM.register_controller(v, v_pid)
    SYSTEM.register_controller(w, w_pid)

    X0 = np.array([500.0, 500.0, 0.0])
    # ROBOT_LQR = Robot(name="lqr_robot", sys=SYSTEM, x=X0, dt=DT, color=(255, 0, 0))
    ROBOT_PID = Robot(name="pid_robot", sys=SYSTEM, x=X0, dt=DT, color=(255, 255, 255))

    SIM = Simulation(t=100, fps=FPS, reference=REFERENCE_GENERATOR)

    # SIM.register_robot(ROBOT_LQR)
    SIM.register_robot(ROBOT_PID)

    asyncio.run(SIM.run_sim())
