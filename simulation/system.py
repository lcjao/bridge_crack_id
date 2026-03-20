"""
车-桥耦合仿真系统核心模块
Bridge-Vehicle Coupled System Simulation
"""

import numpy as np
from scipy import linalg
import math


class BridgeVehicleSystem:
    """
    车-桥耦合系统类

    用于模拟移动车辆通过桥梁时的动力学响应，支持：
    - 健康状态和损伤状态分析
    - 裂纹损伤建模（刚度衰减）
    - 路面粗糙度生成（ISO 8608标准）
    - CPDV（接触点位移变化）计算
    """

    def __init__(self, params=None):
        """
        初始化车-桥耦合系统

        Args:
            params: 参数字典，若为None则使用默认参数
        """
        # 默认参数
        self.params = params or {}

        # 确保数值参数为正确类型
        def to_float(val, default):
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return float(val)
            return default

        def to_int(val, default):
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return int(val)
            return default

        # 路面等级参数
        self.road_type = self.params.get("road_type", "a")  # A级路面

        # 车辆参数
        self.mv = to_float(self.params.get("mv", 1000), 1000)  # 车辆质量 (kg)
        self.kv = to_float(self.params.get("kv", 170000), 170000)  # 车辆悬架刚度 (N/m)
        self.cv = to_float(
            self.params.get("cv", 1000), 1000
        )  # 车辆悬架阻尼系数 (N/(m/s))
        self.V = to_float(self.params.get("V", 10), 10)  # 车辆速度 (m/s)
        self.g = 9.8  # 重力加速度 (m/s^2)
        self.fv = np.sqrt(self.kv / self.mv) / (2 * np.pi)  # 车辆固有频率 (Hz)

        # 桥梁参数
        self.L = to_float(self.params.get("L", 30), 30)  # 桥梁长度 (m)
        self.E = to_float(self.params.get("E", 2.75e10), 2.75e10)  # 桥梁弹性模量 (Pa)
        self.I = to_float(self.params.get("I", 0.175), 0.175)  # 桥梁惯性矩 (m^4)
        self.m = to_float(self.params.get("m", 1000), 1000)  # 单位长度质量 (kg/m)
        self.EL = to_int(self.params.get("EL", 30), 30)  # 桥梁离散单元数量
        self.depth = to_float(self.params.get("depth", 1.0), 1.0)  # 梁深 (m)
        self.width = to_float(self.params.get("width", 0.3), 0.3)  # 梁宽 (m)

        # 模态参数
        self.n_modes = to_int(self.params.get("n_modes", 4), 4)  # 模态数量
        self.kexi = to_float(self.params.get("kexi", 0.02), 0.02)  # 阻尼比

        # 分析参数
        self.ttotal = self.L / self.V  # 总分析时间 (s)
        self.deltat = to_float(self.params.get("deltat", 0.01), 0.01)  # 时间步长 (s)
        self.t = np.arange(0, self.ttotal + self.deltat, self.deltat)  # 时间向量
        self.tstep = len(self.t)

        # Newmark-β法参数
        self.gamma = to_float(self.params.get("gamma", 0.5), 0.5)
        self.beta = to_float(self.params.get("beta", 0.25), 0.25)
        self._setup_newmark_params()

        # 裂纹损伤参数
        self.crack_position = 0  # 裂纹位置 (m)
        self.crack_depth_ratio = 0  # 裂纹深度比 (0-1)
        self.crack_width = 0.3  # 裂纹宽度 (m)
        self.has_crack = False  # 是否存在裂纹

        # 存储结果
        self.results = {}
        self.healthy_results = {}
        self.damaged_results = {}
        self.uc = []  # 接触点位移响应
        self.CPDV = []  # 接触点位移变化
        self.uc_healthy = None  # 健康状态接触点位移

        # 随机种子
        self.rng = np.random.RandomState(42)

    def _setup_newmark_params(self):
        """设置Newmark-β法参数"""
        dt = self.deltat
        beta = self.beta
        gamma = self.gamma

        self.Alpha_0 = 1 / (beta * dt**2)
        self.Alpha_1 = gamma / (beta * dt)
        self.Alpha_2 = 1 / (beta * dt)
        self.Alpha_3 = 1 / (2 * beta) - 1
        self.Alpha_4 = gamma / beta - 1
        self.Alpha_5 = dt / 2 * (gamma / beta - 2)
        self.Alpha_6 = dt * (1 - gamma)
        self.Alpha_7 = gamma * dt

    def beam_km0_f(self, m, L, E, I, EL):
        """
        桥梁有限元质量矩阵和刚度矩阵生成函数

        Args:
            m: 单位长度质量 (kg/m)
            L: 桥梁长度 (m)
            E: 桥梁弹性模量 (Pa)
            I: 桥梁惯性矩 (m^4)
            EL: 桥梁离散单元数量

        Returns:
            M_global: 全局质量矩阵
            K_global: 全局刚度矩阵
        """
        Le = L / EL  # 单元长度

        # 单元质量矩阵 (一致质量矩阵)
        mb = (m * Le / 420) * np.array(
            [
                [156, 22 * Le, 54, -13 * Le],
                [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
                [54, 13 * Le, 156, 22 * Le],
                [-13 * Le, -3 * Le**2, 22 * Le, 4 * Le**2],
            ]
        )

        # 单元刚度矩阵
        kb = (E * I / Le**3) * np.array(
            [
                [12, 6 * Le, -12, 6 * Le],
                [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                [-12, -6 * Le, 12, -6 * Le],
                [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
            ]
        )

        n_dof = 2 * (EL + 1)
        K_global = np.zeros((n_dof, n_dof))
        M_global = np.zeros((n_dof, n_dof))

        # 裂纹损伤处理
        crack_elem = None
        K_cj = None

        if self.has_crack:
            # 确定裂纹所在单元
            crack_elem = int(np.floor(self.crack_position / Le)) + 1
            if crack_elem > EL:
                crack_elem = EL
            xi_j = self.crack_position - (crack_elem - 1) * Le

            # 计算裂纹引起的惯性矩变化
            d = self.depth
            w = self.crack_width
            I_0 = w * d**3 / 12
            d_cj = self.crack_depth_ratio * d
            I_cj = w * (d - d_cj) ** 3 / 12

            # 刚度降低区域的特征长度
            l_c = 1.5 * self.depth
            xi_j_norm = xi_j / Le
            l_c_norm = l_c / Le

            # 计算刚度降低矩阵元素
            C = (I_0 - I_cj) / I_cj

            k11 = (12 * E * (I_0 - I_cj) / Le**4) * (
                (2 * l_c**3) / (Le**2) + 3 * l_c * ((2 * xi_j_norm - 1) ** 2)
            )
            k12 = (12 * E * (I_0 - I_cj) / Le**3) * (
                (l_c**3) / (Le**2) + l_c * (2 - 7 * xi_j_norm + 6 * xi_j_norm**2)
            )
            k14 = (12 * E * (I_0 - I_cj) / Le**3) * (
                (l_c**3) / (Le**2) + l_c * (1 - 5 * xi_j_norm + 6 * xi_j_norm**2)
            )
            k22 = (12 * E * (I_0 - I_cj) / Le**2) * (
                (3 * l_c**3) / (Le**2) + 2 * l_c * ((3 * xi_j_norm - 2) ** 2)
            )
            k24 = (12 * E * (I_0 - I_cj) / Le**2) * (
                (3 * l_c**3) / (Le**2)
                + 2 * l_c * (2 - 9 * xi_j_norm + 9 * xi_j_norm**2)
            )
            k44 = (12 * E * (I_0 - I_cj) / Le**2) * (
                (3 * l_c**3) / (Le**2) + 2 * l_c * ((3 * xi_j_norm - 1) ** 2)
            )

            # 组装裂纹单元的刚度降低矩阵
            K_cj = np.array(
                [
                    [k11, k12, -k11, k14],
                    [k12, k22, -k12, k24],
                    [-k11, -k12, k11, -k14],
                    [k14, k24, -k14, k44],
                ]
            )

        # 组装全局矩阵
        for i in range(EL):
            K_e = kb.copy()

            # 应用裂纹刚度降低
            if self.has_crack and (i + 1) == crack_elem:
                K_e_crack = K_e - K_cj
                K_e_crack = (K_e_crack + K_e_crack.T) / 2
                K_global[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += K_e_crack
            else:
                K_global[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += K_e

            M_global[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += mb

        # 应用边界条件（简支梁）
        keep_indices = list(range(n_dof))
        keep_indices.remove(0)
        keep_indices.remove(n_dof - 2)
        M_global = M_global[np.ix_(keep_indices, keep_indices)]
        K_global = K_global[np.ix_(keep_indices, keep_indices)]

        return M_global, K_global

    def psd_r(self, roadtype, L, V, deltat):
        """
        路面不平顺度生成函数（ISO 8608标准）

        Args:
            roadtype: 路面等级 'a', 'b', 'c'
            L: 桥梁长度 (m)
            V: 车辆速度 (m/s)
            deltat: 时间步长 (s)

        Returns:
            RX: 路面不平顺序列
            dRX: 路面不平顺一阶导数
        """
        ts = 5 * deltat

        # A-C级路面PSD系数
        road_dict = {
            "a": 0.001 * 1e-6,
            "b": 8 * 1e-6,
            "c": 16 * 1e-6,
        }
        Gd_n0 = road_dict.get(roadtype.lower(), 0.001 * 1e-6)

        n_max = 2.83
        n_min = 0.011
        n0 = 0.1
        N = max(100, int(L / V / ts))

        t_total = L / V
        num_points = int(t_total / deltat) + 1
        t = np.linspace(0, t_total, num_points)

        # 随机相位角
        theta = self.rng.uniform(0, 2 * np.pi, N)

        # 空间频率序列
        delta_n = (n_max - n_min) / N
        n_k = n_min + np.arange(N) * delta_n

        # 功率谱密度
        Gd_n = Gd_n0 * (n_k / n0) ** (-2)

        # 计算路面不平顺
        RX = np.zeros(num_points)
        dRX = np.zeros(num_points)
        amplitude = np.sqrt(2 * Gd_n * delta_n)

        for i in range(num_points):
            phase = 2 * np.pi * n_k * V * t[i] + theta
            RX[i] = np.sum(amplitude * np.cos(phase))
            dRX[i] = -2 * np.pi * V * np.sum(n_k * amplitude * np.sin(phase))

        # 长度匹配
        if len(RX) > self.tstep:
            return RX[: self.tstep], dRX[: self.tstep]
        else:
            RX_full = np.zeros(self.tstep)
            dRX_full = np.zeros(self.tstep)
            RX_full[: len(RX)] = RX
            dRX_full[: len(dRX)] = dRX
            return RX_full, dRX_full

    def force_vector(self, EL, tstep, V, deltat, L):
        """
        形函数矩阵和荷载向量

        Args:
            EL: 单元数量
            tstep: 时间步数
            V: 车速 (m/s)
            deltat: 时间步长 (s)
            L: 桥梁长度 (m)

        Returns:
            REXT: 荷载向量
            dREXT: 荷载向量一阶导数
        """
        Le = L / EL
        F = np.zeros((EL + 1, tstep))
        MO = np.zeros((EL + 1, tstep))
        df = np.zeros((EL + 1, tstep))
        dMO = np.zeros((EL + 1, tstep))

        for i in range(tstep):
            xp = V * i * deltat

            if xp >= L:
                continue

            s = math.ceil(xp / Le)
            if s > EL:
                s = EL
            xc = xp - (s - 1) * Le

            if Le > 0:
                zeta = xc / Le
                N1 = 1 - 3 * zeta**2 + 2 * zeta**3
                N2 = Le * (zeta - 2 * zeta**2 + zeta**3)
                N3 = 3 * zeta**2 - 2 * zeta**3
                N4 = Le * (-(zeta**2) + zeta**3)

                dN1_dx = (-6 * zeta + 6 * zeta**2) / Le
                dN2_dx = 1 - 4 * zeta + 3 * zeta**2
                dN3_dx = (6 * zeta - 6 * zeta**2) / Le
                dN4_dx = -2 * zeta + 3 * zeta**2

            if s <= EL:
                F[s - 1, i] = N1
                MO[s - 1, i] = N2
                F[s, i] = N3
                MO[s, i] = N4
                df[s - 1, i] = dN1_dx
                df[s, i] = dN2_dx
                dMO[s - 1, i] = dN3_dx
                dMO[s, i] = dN4_dx

        REXT2 = np.zeros((2 * (EL + 1), tstep))
        dREXT2 = np.zeros((2 * (EL + 1), tstep))

        for ni in range(EL + 1):
            REXT2[2 * ni, :] = F[ni, :]
            REXT2[2 * ni + 1, :] = MO[ni, :]
            dREXT2[2 * ni, :] = df[ni, :]
            dREXT2[2 * ni + 1, :] = dMO[ni, :]

        # 应用边界条件
        keep_indices = list(range(2 * (EL + 1)))
        keep_indices.remove(0)
        keep_indices.remove(2 * (EL + 1) - 2)
        REXT = REXT2[keep_indices, :]
        dREXT = dREXT2[keep_indices, :]

        return REXT, dREXT

    def analyze(self):
        """
        执行车-桥耦合分析
        """
        # 桥梁有限元模型
        self.M, self.K = self.beam_km0_f(self.m, self.L, self.E, self.I, self.EL)

        # 确保矩阵对称
        self.K = (self.K + self.K.T) / 2
        self.M = (self.M + self.M.T) / 2

        # 添加正则化防止数值问题
        K_reg = self.K + 1e-6 * np.eye(self.K.shape[0])
        M_reg = self.M + 1e-6 * np.eye(self.M.shape[0])

        # 特征值分析
        if np.all(np.linalg.eigvals(K_reg) > 0) and np.all(
            np.linalg.eigvals(M_reg) > 0
        ):
            eigvals, eigvecs = linalg.eig(K_reg, M_reg)
            real_mask = np.isreal(eigvals)

            if np.sum(real_mask) >= self.n_modes:
                eigvals = np.real(eigvals[real_mask])
                eigvecs = np.real(eigvecs[:, real_mask])

                sorted_indices = np.argsort(eigvals)
                eigvals = eigvals[sorted_indices]
                eigvecs = eigvecs[:, sorted_indices]

                self.phi = eigvecs[:, : self.n_modes]
                self.Omega = np.sqrt(eigvals[: self.n_modes])
                self.Frequ = self.Omega / (2 * np.pi)
            else:
                self.Frequ = self._compute_theoretical_freq()
        else:
            self.Frequ = self._compute_theoretical_freq()

        # 瑞利阻尼
        self._setup_damping()

        # 路面激励
        self.RX, self.dRX = self.psd_r(self.road_type, self.L, self.V, self.deltat)

        # 力向量
        self.REXT, self.dREXT = self.force_vector(
            self.EL, self.tstep, self.V, self.deltat, self.L
        )

        # 车桥耦合分析
        self._solve_dynamic()

        # 提取响应
        self.zv_DIS = self.u[0, :]
        self.zv_VEL = self.du[0, :]
        self.zv_ACC = self.ddu[0, :]

        # 接触点位移响应
        self.uc = self.zv_DIS + self.zv_ACC / ((2 * np.pi * self.fv) ** 2 + 1e-10)

        # 数值稳定性处理 - 裁剪异常值
        self.uc = np.clip(self.uc, -1e6, 1e6)
        self.uc = np.nan_to_num(self.uc, nan=0.0, posinf=1e6, neginf=-1e6)

        # 桥梁位移响应
        N = int(np.ceil(self.tstep / 2)) - 1
        self.uc_mid = np.zeros(self.tstep)
        for k in range(self.tstep):
            if k < self.u.shape[1] and N < self.REXT.shape[1]:
                if not np.any(np.isnan(self.REXT[:, N])) and not np.any(
                    np.isnan(self.u[1:, k])
                ):
                    self.uc_mid[k] = self.REXT[:, N].T @ self.u[1:, k]

    def _compute_theoretical_freq(self):
        """计算理论频率"""
        freq = np.zeros(self.n_modes)
        for n in range(1, self.n_modes + 1):
            freq[n - 1] = (
                (n**2) / (2 * self.L**2) * np.sqrt(self.E * self.I / (np.pi * self.m))
            )
        return freq

    def _setup_damping(self):
        """设置瑞利阻尼"""
        if len(self.Frequ) >= 2:
            f_1 = self.Frequ[0]
            f_2 = self.Frequ[1]
            w1 = 2 * np.pi * f_1
            w2 = 2 * np.pi * f_2
            coefficient = 2 * w1 * w2 / (w1**2 - w2**2)
            matrix = np.array([[w1, -w2], [-1 / w1, 1 / w2]])
            a0, a1 = np.dot(matrix, [self.kexi, self.kexi]) * coefficient
            self.C = a0 * self.M + a1 * self.K

    def _solve_dynamic(self):
        """Newmark-β法时程积分"""
        Ndof = self.M.shape[0]
        total_dof = Ndof + 1

        self.u = np.zeros((total_dof, self.tstep))
        self.du = np.zeros((total_dof, self.tstep))
        self.ddu = np.zeros((total_dof, self.tstep))

        mv_matrix = np.array([[self.mv]])
        kv_matrix = np.array([[self.kv]])
        cv_matrix = np.array([[self.cv]])

        for k in range(1, self.tstep):
            if k >= self.REXT.shape[1]:
                break

            L1 = self.REXT[:, k].reshape(-1, 1)
            dL1 = self.dREXT[:, k].reshape(-1, 1)

            r1 = self.RX[k] if k < len(self.RX) else 0
            dr1 = self.dRX[k] if k < len(self.dRX) else 0

            # 组装系统矩阵
            Mn_top = np.hstack([mv_matrix, np.zeros((1, Ndof))])
            Mn_bottom = np.hstack([np.zeros((Ndof, 1)), self.M])
            Mn = np.vstack([Mn_top, Mn_bottom])

            cv_L1 = self.cv * L1
            Cn_top = np.hstack([cv_matrix, -cv_L1.T])
            Cn_bottom = np.hstack([-cv_L1, self.C + self.cv * (L1 @ L1.T)])
            Cn = np.vstack([Cn_top, Cn_bottom])

            kv_L1 = self.kv * L1
            cv_V_dL1 = self.cv * self.V * dL1
            Kn_top = np.hstack([kv_matrix, -cv_V_dL1.T - kv_L1.T])
            Kn_bottom = np.hstack(
                [
                    -kv_L1,
                    self.K + self.cv * self.V * (L1 @ dL1.T) + self.kv * (L1 @ L1.T),
                ]
            )
            Kn = np.vstack([Kn_top, Kn_bottom])

            Fn_top = np.array([[self.cv * self.V * dr1 + self.kv * r1]])
            Fn_bottom = (
                -self.cv * self.V * dr1 * L1 - self.kv * r1 * L1 - self.mv * self.g * L1
            )
            Fn = np.vstack([Fn_top, Fn_bottom])

            # Newmark-β法求解
            try:
                u_prev = self.u[:, k - 1].reshape(-1, 1)
                du_prev = self.du[:, k - 1].reshape(-1, 1)
                ddu_prev = self.ddu[:, k - 1].reshape(-1, 1)

                Keff = Kn + self.Alpha_0 * Mn + self.Alpha_1 * Cn

                if np.any(np.isnan(Keff)) or np.any(np.isinf(Keff)):
                    raise ValueError("Keff contains NaN or inf")

                term1 = (
                    self.Alpha_0 * u_prev
                    + self.Alpha_2 * du_prev
                    + self.Alpha_3 * ddu_prev
                )
                term2 = (
                    self.Alpha_1 * u_prev
                    + self.Alpha_4 * du_prev
                    + self.Alpha_5 * ddu_prev
                )

                if np.any(np.isnan(term1)) or np.any(np.isinf(term1)):
                    raise ValueError("term1 contains invalid values")

                Feff = Fn + Mn @ term1 + Cn @ term2
                u_new = linalg.solve(Keff, Feff)
                self.u[:, k : k + 1] = u_new

                self.ddu[:, k : k + 1] = (
                    self.Alpha_0 * (u_new - u_prev)
                    - self.Alpha_2 * du_prev
                    - self.Alpha_3 * ddu_prev
                )
                self.du[:, k : k + 1] = (
                    du_prev
                    + self.Alpha_6 * ddu_prev
                    + self.Alpha_7 * self.ddu[:, k : k + 1]
                )

                if np.any(np.isnan(self.u[:, k])) or np.any(np.isinf(self.u[:, k])):
                    raise ValueError("Displacement contains invalid values")

            except Exception:
                self.u[:, k] = self.u[:, k - 1]
                self.du[:, k] = self.du[:, k - 1]
                self.ddu[:, k] = self.ddu[:, k - 1]

    def run_analysis(self):
        """
        运行完整分析流程：健康状态 -> 损伤状态 -> CPDV
        """
        # 健康状态分析
        self.has_crack = False
        self.analyze()
        self.healthy_results = {
            "zv_DIS": self.zv_DIS.copy(),
            "uc_mid": self.uc_mid.copy(),
            "Frequ": self.Frequ.copy(),
            "K_matrix": self.K.copy(),
            "uc": self.uc.copy(),
        }
        self.uc_healthy = self.uc.copy()

        return self.healthy_results

    def analyze_damage(self, crack_position, crack_depth_ratio):
        """
        分析指定损伤状态

        Args:
            crack_position: 裂纹位置 (m)
            crack_depth_ratio: 裂纹深度比 (0-1)

        Returns:
            损伤状态结果字典
        """
        self.crack_position = crack_position
        self.crack_depth_ratio = crack_depth_ratio
        self.has_crack = True

        # 重新生成刚度矩阵
        self.M, self.K = self.beam_km0_f(self.m, self.L, self.E, self.I, self.EL)

        # 分析
        self.analyze()

        return {
            "zv_DIS": self.zv_DIS.copy(),
            "uc_mid": self.uc_mid.copy(),
            "Frequ": self.Frequ.copy(),
            "K_matrix": self.K.copy(),
            "uc": self.uc.copy(),
        }

    def calculate_cpdv(self, damaged_uc):
        """
        计算CPDV（接触点位移变化）

        Args:
            damaged_uc: 损伤状态接触点位移

        Returns:
            CPDV序列
        """
        if self.uc_healthy is None:
            raise ValueError("请先运行健康状态分析")

        # 计算CPDV并添加数值稳定性处理
        cpdv = self.uc_healthy - damaged_uc
        # 使用更合理的裁剪范围 (±1米)
        cpdv = np.clip(cpdv, -1.0, 1.0)
        cpdv = np.nan_to_num(cpdv, nan=0.0, posinf=0.0, neginf=0.0)

        return cpdv
