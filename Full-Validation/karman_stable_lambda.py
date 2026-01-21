# -*- coding: utf-8 -*-
"""
卡门涡街模拟器 - CFUT-NS 唯象模型
基于 NS++.md 理论：F_topo = -α·ρ·u，α = ν_topo / l_c²

物理哲学：
  传统观点：α=0 是正常，α>0 是修正
  CFUT真相：α>0 是常态（现实有边界），α=0 是特例（理想化）

  α=0.000000：无限大理想空间（纯理论）
  α=0.00001：实验室光滑管道（现实最优）⭐ 本模拟
  α=0.0001：普通工程管道（可见效应）
  α=0.001：多孔介质（强拓扑约束）
  α=8-82 s⁻²：地质流体（NS++.md，岩浆在裂隙）

核心洞见：现实世界本质上是拓扑复杂的！

注：α（拓扑阻尼系数）≠ λ（完整模型中的无量纲耦合系数）
    本模拟使用唯象简化版，直接调节 α
"""

import pygame
import numpy as np
import sys
from numba import jit

pygame.init()

# 窗口参数
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 400
NX = 300
NY = 100
SCALE = 4

# 物理常数与参数（基于 NS++.md 和 CFUT 框架）
# 唯象模型：F_topo = -α·ρ·u，其中 α = ν_topo / l_c²

class SimParams:
    def __init__(self):
        self.inlet_velocity = 0.08
        self.viscosity = 0.015
        self.cfut_on = False
        self.cylinder_radius = 12

        # 拓扑物理参数（基于 NS++.md 公式 α = ν_topo/l_c²）
        self.alpha_topo = 0.0  # 拓扑阻尼系数 [s⁻²]

        # 调节步长
        self.vel_step = 0.01
        self.visc_step = 0.005
        self.alpha_step = 0.00001  # α 步长（物理范围 10^-5 级别）

        # 安全限制（基于实际物理标定）
        self.alpha_max = 0.001     # α 上限 [s⁻²]
        self.alpha_warning = 0.0001  # 警告阈值 [s⁻²]

    @property
    def reynolds(self):
        return self.inlet_velocity * 2 * self.cylinder_radius / self.viscosity

    @property
    def characteristic_length(self):
        """拓扑相干长度 l_c（特征尺度）"""
        # 对于圆柱绕流，使用圆柱直径作为特征尺度
        return 2 * self.cylinder_radius * (WINDOW_WIDTH / NX / SCALE)  # 转换为物理单位 [m]

    @property
    def nu_topo(self):
        """拓扑黏度 ν_topo = α·l_c² [m²/s]"""
        l_c = self.characteristic_length
        return self.alpha_topo * l_c * l_c

    @property
    def alpha_effective(self):
        """有效阻尼系数（与 α_topo 相同，保持接口兼容）"""
        return self.alpha_topo

params = SimParams()

CYLINDER_X = 60
CYLINDER_Y = NY // 2

# LBM 参数
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


@jit(nopython=True, cache=True)
def feq(rho, ux, uy):
    """平衡分布"""
    f = np.zeros((9, NY, NX))
    usqr = ux*ux + uy*uy
    for i in range(9):
        cu = CX[i]*ux + CY[i]*uy
        f[i] = rho * W[i] * (1 + 3*cu + 4.5*cu*cu - 1.5*usqr)
    return f


@jit(nopython=True, cache=True)
def compute_vorticity(ux, uy):
    """计算涡度"""
    omega = np.zeros((NY, NX))
    for j in range(1, NY-1):
        for i in range(1, NX-1):
            duy_dx = (uy[j, i+1] - uy[j, i-1]) * 0.5
            dux_dy = (ux[j+1, i] - ux[j-1, i]) * 0.5
            omega[j, i] = duy_dx - dux_dy
    return omega


class KarmanSim:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Karman Vortex - CFUT-NS Phenomenological Model")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 19)
        self.big_font = pygame.font.Font(None, 36)

        self.running = True
        self.paused = False
        self.show_vort = True

        self.reset()

    def create_obstacle(self):
        """创建圆柱"""
        obst = np.zeros((NY, NX), dtype=bool)
        for j in range(NY):
            for i in range(NX):
                if (i-CYLINDER_X)**2 + (j-CYLINDER_Y)**2 <= params.cylinder_radius**2:
                    obst[j,i] = True
        return obst

    def reset(self):
        """初始化"""
        self.rho = np.ones((NY, NX))
        self.ux = np.ones((NY, NX)) * params.inlet_velocity
        self.uy = np.zeros((NY, NX))

        # 初始扰动
        np.random.seed(42)
        self.uy += np.random.randn(NY, NX) * 0.01 * params.inlet_velocity

        self.f = feq(self.rho, self.ux, self.uy)
        self.tau = 3*params.viscosity + 0.5
        self.omega_lbm = 1.0 / self.tau
        self.obst = self.create_obstacle()

        print(f"[Reset] Re={params.reynolds:.1f}, alpha={params.alpha_topo:.6f} s^-2, l_c={params.characteristic_length:.3f} m, nu_topo={params.nu_topo:.2e} m^2/s")

    def step(self):
        """时间步进"""
        self.tau = 3*params.viscosity + 0.5
        self.omega_lbm = 1.0 / self.tau

        # 流动
        for i in range(9):
            self.f[i] = np.roll(self.f[i], CX[i], axis=1)
            self.f[i] = np.roll(self.f[i], CY[i], axis=0)

        # 边界：入流
        vel = params.inlet_velocity
        self.ux[:,0] = vel
        self.uy[:,0] = 0
        r = 1.0
        self.rho[:,0] = r
        self.f[1,:,0] = self.f[3,:,0] + 2/3*r*vel
        self.f[5,:,0] = self.f[7,:,0] + 1/6*r*vel - 0.5*(self.f[2,:,0]-self.f[4,:,0])
        self.f[8,:,0] = self.f[6,:,0] + 1/6*r*vel + 0.5*(self.f[2,:,0]-self.f[4,:,0])

        # 出流
        self.f[:,:,-1] = self.f[:,:,-2]

        # 障碍物
        for i in range(9):
            self.f[i, self.obst] = self.f[OPP[i], self.obst]

        # 宏观量
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.clip(self.rho, 0.1, 10.0)
        self.ux = np.sum(self.f * CX.reshape(9,1,1), axis=0) / self.rho
        self.uy = np.sum(self.f * CY.reshape(9,1,1), axis=0) / self.rho

        # 速度限制
        speed = np.sqrt(self.ux**2 + self.uy**2)
        max_speed = 0.3
        mask = speed > max_speed
        if np.any(mask):
            self.ux[mask] *= max_speed / speed[mask]
            self.uy[mask] *= max_speed / speed[mask]

        # ==== CFUT 唯象拓扑阻尼 ====
        if params.cfut_on and params.alpha_effective > 0:
            # 简单而稳定的唯象阻尼：u_new = u / (1 + α)
            damping_factor = 1.0 / (1.0 + params.alpha_effective)
            self.ux *= damping_factor
            self.uy *= damping_factor

        # 碰撞
        feq_val = feq(self.rho, self.ux, self.uy)
        self.f += self.omega_lbm * (feq_val - self.f)

        # 数值健康检查
        if np.any(np.isnan(self.f)) or np.any(np.isinf(self.f)) or np.max(speed) > 0.5:
            print("⚠️ Instability detected! Resetting...")
            self.reset()

    def draw(self):
        """渲染"""
        if self.show_vort:
            data = compute_vorticity(self.ux, self.uy)
            vmax = 0.3
            field = np.clip((data + vmax) / (2*vmax), 0, 1)
        else:
            data = np.sqrt(self.ux**2 + self.uy**2)
            field = np.clip(data / (1.5*params.inlet_velocity), 0, 1)

        # 彩色映射
        rgb = np.zeros((NY, NX, 3), dtype=np.uint8)
        rgb[:,:,0] = (np.clip(field * 2.5 - 0.5, 0, 1) * 255).astype(np.uint8)
        rgb[:,:,1] = (np.clip(np.sin(field * np.pi) * 1.2, 0, 1) * 255).astype(np.uint8)
        rgb[:,:,2] = (np.clip((1 - field) * 1.5, 0, 1) * 255).astype(np.uint8)
        rgb[self.obst] = [255, 255, 255]

        surf = pygame.surfarray.make_surface(rgb.transpose(1,0,2))
        surf = pygame.transform.scale(surf, (WINDOW_WIDTH, WINDOW_HEIGHT))
        self.screen.blit(surf, (0,0))

        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        """UI"""
        # 稳定性状态
        if params.alpha_topo > params.alpha_max:
            stability = "LIMIT!"
            stab_color = (255, 50, 50)
        elif params.alpha_topo > params.alpha_warning:
            stability = "⚠️ High"
            stab_color = (255, 200, 50)
        else:
            stability = "✓ OK"
            stab_color = (50, 255, 50)

        lines = [
            f"Reynolds: {params.reynolds:.1f}",
            f"Velocity: {params.inlet_velocity:.3f}  [W/S]",
            f"Viscosity: {params.viscosity:.4f}  [A/D]",
            "",
            f"Alpha: {params.alpha_topo:.6f} s^-2 {'[ON]' if params.cfut_on else '[OFF]'}  [UP/DN]",
            f"l_c: {params.characteristic_length:.4f} m  (topo coherence length)",
            f"nu_topo: {params.nu_topo:.2e} m^2/s  (= alpha * l_c^2)",
            f"Stability: {stability}  (max={params.alpha_max:.6f})",
            f"Step: {params.alpha_step:.6f}",
            "",
            f"View: {'Vorticity' if self.show_vort else 'Velocity'}",
            f"FPS: {int(self.clock.get_fps())}",
            "",
            "CFUT-NS Phenomenological Model:",
            "F_topo = -alpha * rho * u",
            "alpha = nu_topo / l_c^2",
            "",
            "[T] Toggle  [V] View",
            "[R] Reset   [SPACE] Pause",
        ]

        y = 5
        for i, line in enumerate(lines):
            if "Stability:" in line:
                color = stab_color
            else:
                color = (255, 255, 50)

            txt = self.font.render(line, True, color)
            bg = pygame.Surface((txt.get_width()+10, 21))
            bg.set_alpha(200)
            bg.fill((0, 0, 0))
            self.screen.blit(bg, (5, y))
            self.screen.blit(txt, (9, y+1))
            y += 20

        # 模式指示
        mode_text = "CFUT Damping" if params.cfut_on else "Classical NS"
        mode_color = (255, 100, 100) if params.cfut_on else (100, 255, 100)

        mode_surf = self.big_font.render(mode_text, True, mode_color)
        mode_rect = mode_surf.get_rect(center=(WINDOW_WIDTH//2, 25))

        bg = pygame.Surface((mode_surf.get_width()+20, 40))
        bg.set_alpha(220)
        bg.fill((0, 0, 0))
        bg_rect = bg.get_rect(center=(WINDOW_WIDTH//2, 25))
        self.screen.blit(bg, bg_rect)
        self.screen.blit(mode_surf, mode_rect)

    def events(self):
        """事件处理"""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_v:
                    self.show_vort = not self.show_vort
                elif e.key == pygame.K_r:
                    self.reset()
                elif e.key == pygame.K_t:
                    params.cfut_on = not params.cfut_on
                    print(f"CFUT: {'ON' if params.cfut_on else 'OFF'}")

                # 速度
                elif e.key == pygame.K_w:
                    params.inlet_velocity = min(0.15, params.inlet_velocity + params.vel_step)
                    print(f"Velocity: {params.inlet_velocity:.3f}, Re: {params.reynolds:.1f}")
                elif e.key == pygame.K_s:
                    params.inlet_velocity = max(0.02, params.inlet_velocity - params.vel_step)
                    print(f"Velocity: {params.inlet_velocity:.3f}, Re: {params.reynolds:.1f}")

                # 粘度
                elif e.key == pygame.K_d:
                    params.viscosity = min(0.05, params.viscosity + params.visc_step)
                    print(f"Viscosity: {params.viscosity:.4f}, Re: {params.reynolds:.1f}")
                elif e.key == pygame.K_a:
                    params.viscosity = max(0.005, params.viscosity - params.visc_step)
                    print(f"Viscosity: {params.viscosity:.4f}, Re: {params.reynolds:.1f}")

                # Alpha 调节（拓扑阻尼系数）
                elif e.key == pygame.K_UP:
                    new_alpha = params.alpha_topo + params.alpha_step
                    if new_alpha > params.alpha_max:
                        print(f"Warning: Alpha limit: {params.alpha_max:.6f} s^-2")
                    params.alpha_topo = min(params.alpha_max, new_alpha)
                    print(f"alpha={params.alpha_topo:.6f} s^-2, l_c={params.characteristic_length:.4f} m, nu_topo={params.nu_topo:.2e} m^2/s")

                elif e.key == pygame.K_DOWN:
                    params.alpha_topo = max(0.0, params.alpha_topo - params.alpha_step)
                    print(f"alpha={params.alpha_topo:.6f} s^-2, l_c={params.characteristic_length:.4f} m, nu_topo={params.nu_topo:.2e} m^2/s")

    def run(self):
        """主循环"""
        print("="*60)
        print("Karman Vortex - CFUT-NS Phenomenological Model")
        print("="*60)
        print("Based on NS++.md: F_topo = -alpha * rho * u")
        print("Physical relation: alpha = nu_topo / l_c^2")
        print(f"Characteristic length l_c = {params.characteristic_length:.4f} m")
        print(f"Initial Re: {params.reynolds:.1f}")
        print("="*60)
        print("Controls:")
        print("  UP/DOWN - Alpha (topo damping coefficient)")
        print("  W/S - Velocity,  A/D - Viscosity")
        print("  T - Toggle CFUT,  V - View mode")
        print("  R - Reset,  SPACE - Pause")
        print("="*60)

        while self.running:
            self.events()
            if not self.paused:
                self.step()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = KarmanSim()
    sim.run()
