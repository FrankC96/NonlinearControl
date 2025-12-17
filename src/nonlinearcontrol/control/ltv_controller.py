import numpy as np
import numpy.typing as npt
import control as ctrl

import matplotlib.pyplot as plt

FloatArray = npt.NDArray[np.float64]


def f(x: FloatArray, u: FloatArray) -> FloatArray:
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


def euler_step(f: callable, x: FloatArray, u: FloatArray, dt: float) -> FloatArray:
    return x + f(x, u) * dt


def wrap(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class LTVController:
    def __init__(self, f: callable, type: str):
        assert type in [
            "lqr",
            "ilqr",
        ], f'Supported type controllers or LQR("lqr") or Integral LQR("ilqr")'
        self.f = f
        self.type = type

    def calculate_jacobians(self, x, u, eps=1e-6):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        n = x.size
        k = u.size

        fxu = self.f(x, u)
        m = fxu.size

        A = np.zeros((m, n))
        B = np.zeros((m, k))

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = self.f(x + dx, u)
            f_minus = self.f(x - dx, u)
            A[:, i] = (f_plus - f_minus) / (2 * eps)

        for j in range(k):
            du = np.zeros(k)
            du[j] = eps
            f_plus = self.f(x, u + du)
            f_minus = self.f(x, u - du)
            B[:, j] = (f_plus - f_minus) / (2 * eps)

        return A, B

    def step(
        self,
        x: FloatArray,
        xref: FloatArray,
        u: FloatArray,
        dt: float,
        Qx: FloatArray = np.diag([500, 500, 1]),
        Qz: FloatArray = np.diag([500, 500]),
        R=np.diag([5, 5]),
    ) -> FloatArray:
        A_ct, B_ct = self.calculate_jacobians(x, u)
        A_d = np.eye(len(x)) + A_ct * dt
        B_d = B_ct * dt
        C_d = np.array([[1, 0, 0], [0, 1, 0]])

        n = A_d.shape[0]
        m = B_d.shape[1]
        p = C_d.shape[0]

        D_d = np.zeros((p, m))

        A_aug = np.block([[A_d, np.zeros((n, p))], [C_d, np.zeros((p, p))]])
        B_aug = np.block([[B_d], [D_d]])

        Q_aug = np.block(
            [
                [Qx, np.zeros([n, p])],
                [np.zeros([p, n]), Qz],
            ]
        )
        R_aug = R

        if self.type == "lqr":
            K, P, eigvals = ctrl.dlqr(A_d, B_d, Qx, R)
            e = (x - xref).copy()
            e[2] = wrap(e[2])
            return u - K @ e
        else:
            K, P, eigvals = ctrl.dlqr(A_aug, B_aug, Q_aug, R_aug)
            e = (x - xref).copy()
            e[2] = wrap(e[2])
            K_x = K[:, : len(x)]
            K_y = K[:, -p:]
            return u - K_y @ (C_d @ x) - K_x @ e


if __name__ == "__main__":
    # Simulation time in seconds
    SIM_TIME = 10

    # Simulation time step
    DT = 1e-3

    # Simulation discrete time steps
    SIM_STEPS = int(SIM_TIME / DT)

    # Initial state
    x0 = np.array([1.0, 1.0, 0.0])

    # Initial input
    u0 = np.array([0.0, 0.0])

    # Number of state, input variables
    nx, nu = x0.shape[0], u0.shape[0]

    X, U = np.empty([SIM_STEPS + 1, nx]), np.empty([SIM_STEPS + 1, nu])
    X[0], U[0] = x0, u0
    uref = u0

    # Reference point
    ref = np.array([10.0, 10.0, 0.0])

    ctr = LTVController(f, type="ilqr")
    for k in range(SIM_STEPS):
        dist = np.linalg.norm(X[k, :2] - ref[:2])
        uref[0] = 2.0 if dist > 1e-1 else 0.0001
        uref[1] = np.arctan2(ref[1] - X[k, 1], ref[0] - X[k, 0])

        U[k + 1] = ctr.step(x=X[k], xref=ref, u=uref, dt=DT)
        X[k + 1] = euler_step(f=f, x=X[k], u=U[k + 1], dt=DT)

    pos_fig, pos_ax = plt.subplots()
    pos_ax.plot(X[:, 0], X[:, 1], "o-")
    pos_ax.grid()

    vars_fig, vars_ax = plt.subplots(3, 2)
    vars_ax = vars_ax.flatten()
    for i in range(nx):
        vars_ax[i].axhline(ref[i], color="r", linestyle="--")
        vars_ax[i].plot(X[:, i])
        vars_ax[i].grid()

    for i in range(nu):
        vars_ax[nx + i].plot(U[:, i])
        vars_ax[nx + i].grid()

    plt.show()
