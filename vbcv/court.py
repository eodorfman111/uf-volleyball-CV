import json

import cv2
import numpy as np

def court_points(length_m=18.0, width_m=9.0, attack_line_m=3.0):
    half = length_m / 2
    return {
        "near_left": (0.0, 0.0),
        "near_right": (width_m, 0.0),
        "far_left": (0.0, length_m),
        "far_right": (width_m, length_m),
        "near_attack_left": (0.0, half - attack_line_m),
        "near_attack_right": (width_m, half - attack_line_m),
        "center_left": (0.0, half),
        "center_right": (width_m, half),
        "far_attack_left": (0.0, half + attack_line_m),
        "far_attack_right": (width_m, half + attack_line_m),
    }


def camera_model(dist, image_size):
    if not dist:
        return None
    w, h = image_size
    f = dist.get("f", float(max(w, h)))
    K = np.array([[f, 0, dist.get("cx", w / 2)], [0, f, dist.get("cy", h / 2)], [0, 0, 1]], dtype=np.float64)
    D = np.array([dist.get("k1", 0.0), dist.get("k2", 0.0), 0.0, 0.0], dtype=np.float64)
    return K, D


def undistort(pts, cam):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if cam is None or len(pts) == 0:
        return pts
    K, D = cam
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def distort(pts, cam):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if cam is None or len(pts) == 0:
        return pts
    K, D = cam
    n = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.zeros(4)).reshape(-1, 2)
    obj = np.concatenate([n, np.ones((len(n), 1))], axis=1)
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
    return out.reshape(-1, 2)


def line_residual(pts):
    pts = np.asarray(pts, dtype=np.float64)
    c = pts - pts.mean(axis=0)
    return float(np.linalg.svd(c, compute_uv=False)[-1] / np.sqrt(len(pts)))


def estimate_k1(lines, image_size, bounds=(-0.6, 0.3)):
    from scipy.optimize import minimize_scalar

    def cost(k1):
        cam = camera_model({"k1": k1}, image_size)
        return sum(line_residual(undistort(l, cam)) ** 2 for l in lines)

    r = minimize_scalar(cost, bounds=bounds, method="bounded")
    return float(r.x), float(np.sqrt(cost(0.0) / len(lines))), float(np.sqrt(r.fun / len(lines)))


def fit_homography(img_pts, court_pts):
    img = np.asarray(img_pts, dtype=np.float64)
    crt = np.asarray(court_pts, dtype=np.float64)
    if len(img) >= 10:
        H, _ = cv2.findHomography(img, crt, cv2.RANSAC, 0.15)
    else:
        H, _ = cv2.findHomography(img, crt, 0)
    return H


def project(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def reprojection_errors(H, img_pts, court_pts):
    return np.linalg.norm(project(H, img_pts) - np.asarray(court_pts, dtype=np.float64), axis=1)


def leave_one_out(img_pts, court_pts):
    img = np.asarray(img_pts, dtype=np.float64)
    crt = np.asarray(court_pts, dtype=np.float64)
    errs = []
    for i in range(len(img)):
        keep = np.arange(len(img)) != i
        if keep.sum() < 4:
            return []
        H = fit_homography(img[keep], crt[keep])
        errs.append(float(reprojection_errors(H, img[i : i + 1], crt[i : i + 1])[0]))
    return errs


def image_to_court(H, pts, cam=None):
    return project(H, undistort(pts, cam))


def court_to_image(H, pts, cam=None):
    return distort(project(np.linalg.inv(H), pts), cam)


def draw_court(frame, H, length_m=18.0, width_m=9.0, attack_line_m=3.0, color=(0, 255, 255), cam=None):
    Hinv = np.linalg.inv(H)
    half = length_m / 2
    lines = [
        ((0, 0), (width_m, 0)),
        ((0, length_m), (width_m, length_m)),
        ((0, 0), (0, length_m)),
        ((width_m, 0), (width_m, length_m)),
        ((0, half), (width_m, half)),
        ((0, half - attack_line_m), (width_m, half - attack_line_m)),
        ((0, half + attack_line_m), (width_m, half + attack_line_m)),
    ]
    out = frame.copy()
    for a, b in lines:
        seg = np.linspace(a, b, 30)
        px = distort(project(Hinv, seg), cam).astype(np.int32)
        cv2.polylines(out, [px], False, color, 2)
    return out


def save_homography(path, H, meta):
    data = dict(meta)
    data["H_img_to_court"] = np.asarray(H).tolist()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_homography(path):
    with open(path) as f:
        data = json.load(f)
    cam = camera_model(data.get("distortion"), data.get("image_size")) if data.get("distortion") else None
    return np.asarray(data["H_img_to_court"]), data, cam
