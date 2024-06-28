# frenet_planning.py

import numpy as np
from copy import deepcopy

def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = 0
    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)
        if dist < min_len:
            min_len = dist
            closest_wp = i
    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x) ** 2 + (y - _y) ** 2)

def get_frenet(x, y, mapx, mapy):
    next_wp = get_closest_waypoints(x, y, mapx, mapy) + 1
    prev_wp = next_wp - 1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y

    frenet_d = get_dist(x_x, x_y, proj_x, proj_y)

    ego_vec = [x - mapx[prev_wp], y - mapy[prev_wp], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec, map_vec)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    frenet_s = 0
    for i in range(prev_wp):
        frenet_s = frenet_s + get_dist(mapx[i], mapy[i], mapx[i + 1], mapy[i + 1])

    frenet_s = frenet_s + get_dist(0, 0, proj_x, proj_y)
    return frenet_s, frenet_d

class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai
        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xf - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vf - self.a1 - 2 * self.a2 * T,
                      af - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_pos(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_vel(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_acc(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

    def calc_jerk(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

class QuarticPolynomial:
    def __init__(self, xi, vi, ai, vf, af, T):
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai
        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vf - self.a1 - 2 * self.a2 * T,
                      af - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_pos(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4

    def calc_vel(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    def calc_acc(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

    def calc_jerk(self, t):
        return 6 * self.a3 + 24 * self.a4 * t

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.c_lat = 0.0
        self.c_lon = 0.0
        self.c_tot = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.kappa = []

def calc_frenet_paths(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d):
    DF_SET = np.array([0.39/2, -0.39/2])  # LANE_WIDTH/2
    frenet_paths = []
    for df in DF_SET:
        for T in np.arange(1, 2 + 0.5, 0.5):
            fp = FrenetPath()
            lat_traj = QuinticPolynomial(di, di_d, di_dd, df, df_d, df_dd, T)
            fp.t = [t for t in np.arange(0.0, T, 0.1)]
            fp.d = [lat_traj.calc_pos(t) for t in fp.t]
            fp.d_d = [lat_traj.calc_vel(t) for t in fp.t]
            fp.d_dd = [lat_traj.calc_acc(t) for t in fp.t]
            fp.d_ddd = [lat_traj.calc_jerk(t) for t in fp.t]

            lon_traj = QuarticPolynomial(si, si_d, si_dd, sf_d, sf_dd, T)
            tfp = deepcopy(fp)
            tfp.s = [lon_traj.calc_pos(t) for t in fp.t]
            tfp.s_d = [lon_traj.calc_vel(t) for t in fp.t]
            tfp.s_dd = [lon_traj.calc_acc(t) for t in fp.t]
            tfp.s_ddd = [lon_traj.calc_jerk(t) for t in fp.t]

            for _t in np.arange(T, 2, 0.1):
                tfp.t.append(_t)
                tfp.d.append(tfp.d[-1])
                _s = tfp.s[-1] + tfp.s_d[-1] * 0.1
                tfp.s.append(_s)
                tfp.s_d.append(tfp.s_d[-1])
                tfp.s_dd.append(tfp.s_dd[-1])
                tfp.s_ddd.append(tfp.s_ddd[-1])
                tfp.d_d.append(tfp.d_d[-1])
                tfp.d_dd.append(tfp.d_dd[-1])
                tfp.d_ddd.append(tfp.d_ddd[-1])

            J_lat = sum(np.power(tfp.d_ddd, 2))
            J_lon = sum(np.power(tfp.s_ddd, 2))
            d_diff = (tfp.d[-1] - opt_d) ** 2
            v_diff = (1 - tfp.s_d[-1]) ** 2

            tfp.c_lat = 0.1 * J_lat + 0.1 * T + 1.0 * d_diff
            tfp.c_lon = 0.1 * J_lon + 0.1 * T + 1.0 * v_diff
            tfp.c_tot = 1.0 * tfp.c_lat + 1.0 * tfp.c_lon
            frenet_paths.append(tfp)
    return frenet_paths

def calc_global_paths(fplist, mapx, mapy, maps):
    for fp in fplist:
        for i in range(len(fp.s)):
            _s = fp.s[i]
            _d = fp.d[i]
            _x, _y, _ = get_cartesian(_s, _d, mapx, mapy, maps)
            fp.x.append(_x)
            fp.y.append(_y)
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])
        for i in range(len(fp.yaw) - 1):
            yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            fp.kappa.append(yaw_diff / fp.ds[i])
    return fplist

def get_cartesian(s, d, mapx, mapy, maps):
    prev_wp = 0
    s = np.mod(s, maps[-2])
    while s > maps[prev_wp + 1] and prev_wp < len(maps) - 2:
        prev_wp += 1
    next_wp = np.mod(prev_wp + 1, len(mapx))
    dx = (mapx[next_wp] - mapx[prev_wp])
    dy = (mapy[next_wp] - mapy[prev_wp])
    heading = np.arctan2(dy, dx)
    seg_s = s - maps[prev_wp]
    seg_x = mapx[prev_wp] + seg_s * np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s * np.sin(heading)
    perp_heading = heading + np.pi / 2
    x = seg_x + d * np.cos(perp_heading)
    y = seg_y + d * np.sin(perp_heading)
    return x, y, heading

def collision_check(fp, obs, mapx, mapy, maps):
    for i in range(len(obs)):
        obs_xy = get_cartesian(obs[i][0], obs[i][1], mapx, mapy, maps)
        d = [((_x - obs_xy[0]) ** 2 + (_y - obs_xy[1]) ** 2) for (_x, _y) in zip(fp.x, fp.y)]
        collision = any([di <= 0.25 ** 2 for di in d])
        if collision:
            return True
    return False

def check_path(fplist, obs, mapx, mapy, maps):
    ok_ind = []
    for i, _path in enumerate(fplist):
        acc_squared = [(abs(a_s ** 2 + a_d ** 2)) for (a_s, a_d) in zip(_path.s_dd, _path.d_dd)]
        if any([v > 2 for v in _path.s_d]):  # Max speed check
            continue
        elif any([acc > 2 ** 2 for acc in acc_squared]):
            continue
        elif any([abs(kappa) > 4 for kappa in fplist[i].kappa]):  # Max curvature check
            continue
        elif collision_check(_path, obs, mapx, mapy, maps):
            continue
        ok_ind.append(i)
    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d):
    fplist = calc_frenet_paths(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d)
    fplist = calc_global_paths(fplist, mapx, mapy, maps)
    fplist = check_path(fplist, obs, mapx, mapy, maps)
    min_cost = float("inf")
    opt_traj = None
    opt_ind = 0
    for fp in fplist:
        if min_cost >= fp.c_tot:
            min_cost = fp.c_tot
            opt_traj = fp
            _opt_ind = opt_ind
        opt_ind += 1
    try:
        _opt_ind
    except NameError:
        print("No solution!")
    return fplist, _opt_ind
