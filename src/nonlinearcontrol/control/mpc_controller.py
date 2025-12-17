from casadi import *
import matplotlib.pyplot as plt

from typing import List, Union

type OCPvar = Union[SX, MX]


class ModelPredictiveController:
    def __init__(
        self,
        n: int,
        Q: DM,
        R: DM,
        dt: float,
        xmin: float = [-inf, -inf, -inf],
        xmax: float = [inf, inf, inf],
        umin: float = [-inf, -inf],
        umax: float = [inf, inf],
    ):
        # Prediction horizon of the problem
        self.n = n
        # State penalization matrix
        if isinstance(Q, np.ndarray):
            self.Q = DM(Q)
            self.nx = self.Q.size1()
        else:
            self.Q = Q
            self.nx = self.Q.shape[0]
        # Input penalization matrix
        if isinstance(R, np.ndarray):
            self.R = DM(R)
            self.nu = self.R.size1()
        else:
            self.R = R
            self.nu = self.R.shape[0]
        # Time-step
        self.dt = dt

        # State variables min bounds
        self.xmin = xmin
        # State variables max bounds
        self.xmax = xmax

        # Input variables min bounds
        self.umin = umin
        # Input variables max bounds
        self.umax = umax

        # FIXME: Fix state and input var buffers, we don't append each var,
        # we replace the final vector each time we append a new one
        # FIXME: MX() might produce too much overhead for simple problems
        self.state_vars: OCPvar = MX()
        self.ref_state_vars: OCPvar = MX()
        self.input_vars: OCPvar = MX()

        self.rhs: OCPvar = MX()
        self.obj: OCPvar = MX()

    def make_step(self, x0: List[float], x_ref: List[float]):
        assert len(x0) == len(
            x_ref
        ), f"State variables should be the same length with the reference state variables."
        assert isinstance(x0, List), f"State vector should be a list."
        assert isinstance(x_ref, List), f"Reference state vector should be a list."

        x_ref = DM(x_ref)

        dae = {
            "x": self.state_vars,
            "p": vertcat(self.input_vars, self.ref_state_vars),
            "ode": self.rhs,
            "quad": self.obj,
        }

        F = integrator("F", "cvodes", dae, 0, self.dt)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym("X0", self.nx)
        w += [Xk]
        lbw += x0
        ubw += x0
        w0 += x0

        # Formulate the NLP
        for k in range(self.n):
            # New NLP variable for the control
            Uk = MX.sym("U_" + str(k), self.nu)
            w += [Uk]
            lbw += self.umin
            ubw += self.umax
            w0 += [0] * self.nu

            # Integrate till the end of the interval
            Fk = F(x0=Xk, p=vertcat(Uk, x_ref))
            Xk_end = Fk["xf"]
            J = J + Fk["qf"]

            # New NLP variable for state at end of interval
            Xk = MX.sym("X_" + str(k + 1), self.nx)
            w += [Xk]
            lbw += self.xmin
            ubw += self.xmax
            w0 += [0] * self.nx

            # Add equality constraint
            g += [Xk_end - Xk]
            lbg += [0] * self.nx
            ubg += [0] * self.nx

        # Terminal constraint for reference tracking
        # g += [Xk_end - x_ref]
        # lbg += [0] * self.nx
        # ubg += [0] * self.nx

        # Create an NLP solver
        prob = {"f": J, "x": vertcat(*w), "g": vertcat(*g)}
        solver = nlpsol(
            "solver",
            "ipopt",
            prob,
            {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"},
        )

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol["x"].full().flatten()

        X = np.empty([self.n + 1, self.nx])
        for i in range(self.nx):
            X[:, i] = w_opt[i :: self.nx + self.nu]

        U = np.empty([self.n, self.nu])
        for i in range(self.nu):
            U[:, i] = w_opt[self.nx + i :: self.nx + self.nu]

        return X, U

    def add_state_variable(self, x: str) -> OCPvar:
        x_var = MX.sym(x)
        x_ref_var = MX.sym(x + "_ref")

        self.state_vars = vertcat(self.state_vars, x_var)
        self.ref_state_vars = vertcat(self.ref_state_vars, x_ref_var)
        return x_var, x_ref_var

    def add_input_variable(self, u: str) -> OCPvar:
        u_var = MX.sym(u)
        self.input_vars = vertcat(self.input_vars, u_var)
        return u_var

    def add_rhs(self, f: OCPvar) -> None:
        self.rhs = vertcat(self.rhs, f)

    def add_cost_func(self, f: OCPvar) -> None:
        self.obj = vertcat(self.obj, f)


def f(x, u, dt):
    return x + dt * np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


if __name__ == "__main__":
    SIM_POINTS = 100
    PRED_HOR = 20
    DT = 0.01
    X0 = [0.0, 0.0, 0.0]
    Q = diag([10000, 10000, 1000])
    R = DM([[1], [20]])

    ocp = ModelPredictiveController(
        PRED_HOR,
        Q,
        R,
        DT,
        xmin=[0.0, 0.0, -1 * np.pi],
        xmax=[1000, 1000, 1 * np.pi],
        umin=[0.0, -10],
        umax=[100, 10],
    )

    x1, x1_ref = ocp.add_state_variable("x1")
    x2, x2_ref = ocp.add_state_variable("x2")
    x3, x3_ref = ocp.add_state_variable("x3")
    u1 = ocp.add_input_variable("u1")
    u2 = ocp.add_input_variable("u2")

    ocp.add_rhs(u1 * cos(x3))
    ocp.add_rhs(u1 * sin(x3))
    ocp.add_rhs(u2)

    x = vertcat(x1, x2, x3)
    x_ref = vertcat(x1_ref, x2_ref, x3_ref)
    u = vertcat(u1, u2)

    ocp.add_cost_func(sum((x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u))

    X = np.empty([SIM_POINTS + 1, x.size1()])
    X[0] = X0
    X_ref = np.empty([SIM_POINTS, x.size1()])
    X[0, :] = X0
    U = np.empty([SIM_POINTS, u.size1()])

    for k in range(SIM_POINTS):
        if k > 1 and np.sum(np.abs(X[k] - X_ref[k - 1])) < 1e1:
            theta = 2 * np.pi * k / SIM_POINTS
        else:
            theta = 2 * np.pi * 0 / SIM_POINTS
        X_ref[k] = [10.0, 10.0, 0.0]

        x_opt, u_opt = ocp.make_step(X[k].tolist(), X_ref[k].tolist())

        U[k] = u_opt[0, :]  # apply first MPC control
        print(
            f"Iteration [{k}] | x [{X[k, 0]:.2f}/ {X_ref[k, 0]:.2f}] | y [{X[k, 1]:.2f}/ {X_ref[k, 1]:.2f}] | u [{U[k, 0]:.2f} | w [{U[k, 1]:.2f}]"
        )

        X[k + 1] = f(X[k], U[k], DT)  # propagate using system dynamics

        plt.cla()
        plt.plot(x_opt[:, 0], x_opt[:, 1], "k-o", label="pred. robot path")
        plt.plot(X[: k + 1, 0], X[: k + 1, 1], "b-o", label="robot path")
        plt.plot(X_ref[k, 0], X_ref[k, 1], "r-o", label="reference path")

        plt.axis("equal")
        plt.pause(0.05)

    plt.show()
