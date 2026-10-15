import numpy as np
from scipy.optimize import least_squares

G = 9.81


def ballistic(params, t):
    x0, y0, z0, vx, vy, vz = params
    return np.stack([x0 + vx * t, y0 + vy * t, z0 + vz * t - G * t * t / 2], 1)


def project(P, X):
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:]


def fit_ballistic(P, t, uv, init=(4.5, -1.0, 2.5, 0.0, 15.0, 0.0)):
    t = np.asarray(t, float) - t[0]
    r = least_squares(lambda q: (project(P, ballistic(q, t)) - uv).ravel(), np.asarray(init, float), loss="soft_l1", f_scale=3.0)
    X = ballistic(r.x, t)
    v0 = float(np.linalg.norm(r.x[3:]))
    avg = float(np.linalg.norm(X[-1] - X[0]) / max(t[-1], 1e-9))
    return {"params": r.x.tolist(), "v0_kmh": v0 * 3.6, "avg_kmh": avg * 3.6, "rms_px": float(np.sqrt(np.mean(r.fun**2)))}


def camera_from_h(H_img_to_court, size):
    Hc = np.linalg.inv(H_img_to_court)
    cx, cy = size[0] / 2, size[1] / 2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    h = T @ Hc
    h1, h2 = h[:, 0], h[:, 1]
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    if f2 <= 0:
        return None
    f = np.sqrt(f2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    A = np.linalg.inv(K) @ Hc
    lam = 1 / np.linalg.norm(A[:, 0])
    r1, r2, t = A[:, 0] * lam, A[:, 1] * lam, A[:, 2] * lam
    r3 = np.cross(r1, r2)
    P = K @ np.column_stack([r1, r2, r3, t])
    if (P @ np.array([4.5, 9.0, 0, 1]))[2] < 0:
        P = K @ np.column_stack([-r1, -r2, r3, -t])
    return {"f_px": float(f), "P": P}
