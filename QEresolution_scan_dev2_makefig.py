import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi, sin, cos, tan, sqrt, arcsin, arccos
from numpy import arctan2  # atan2はarctan2としてインポートする必要があります
import configparser
import os
import sys
from matplotlib.widgets import Slider
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize

from PIL import Image  # GIF 保存のために必要

def make_figure_from_RM(RM,lc_param,rl,calc_params):
    sv1 = lc_param['sv1']
    sv2 = lc_param['sv2']
    sv3 = lc_param['sv3']
    astar = rl["astar"]
    bstar = rl["bstar"]
    cstar = rl["cstar"]
    cph = calc_params["h"]
    cpk = calc_params["k"]
    cpl = calc_params["l"]
    cphw = calc_params["hw"]

    Qx = sv1[0]*astar+sv1[1]*bstar+sv1[2]*cstar
    Qy = sv2[0]*astar+sv2[1]*bstar+sv2[2]*cstar
    Qz = sv3[0]*astar+sv3[1]*bstar+sv3[2]*cstar
    QE_sets = [cphw, cph, cpk, cpl]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2グリッドのサブプロット作成
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.3, hspace=0.4)

    # サブプロットの指定：左上 (0, 0)、右上 (0, 1)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    # 初期タイトル
    plt.suptitle(
        f'ℏω: {QE_sets[0]} meV, h: {QE_sets[1]}, k: {QE_sets[2]}, l: {QE_sets[3]}',
        fontsize=12
    )

    # 制約条件（楕円球の式 = 0 を満たす）
    # 制約条件
    def constraint(params, RM):
        x, y, z, w = params
        return fun4(x, y, z, w, RM)
    
    # 最大値を探索する関数
    # 最大値を探索
    def find_max_along_axis(RM, axis="x"):
        initial_guess = [0, 0, 0, 0]  # 4次元原点
        axis_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        idx = axis_map[axis]

        def objective(params):
            return -params[idx]  # 最大化したいので符号反転

        constraints = {"type": "eq", "fun": constraint, "args": (RM,)}

        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            constraints=constraints,
            options={"disp": False},
        )
        return result.x[idx], result.x  # 軸方向の最大値と座標
    
    # 各軸の最大値を計算
    max_x, coords_x = find_max_along_axis(RM, axis="x")# Q//
    max_y, coords_y = find_max_along_axis(RM, axis="y")# Q⊥
    max_z, coords_z = find_max_along_axis(RM, axis="z")# E
    max_w, coords_w = find_max_along_axis(RM, axis="w")# E
    
    def nice_round(v):
        if v <= 0:
            return 0

        exponent = np.floor(np.log10(v))
        base = v / 10**exponent

        if base <= 1:
            nice_base = 1
        elif base <= 2:
            nice_base = 2
        elif base <= 5:
            nice_base = 5
        else:
            nice_base = 10

        return nice_base * 10**exponent

    Xrange_lim = nice_round(max_x)
    Yrange_lim = nice_round(max_y)
    Zrange_lim = nice_round(max_z)
    Wrange_lim = nice_round(max_w)
    
    '''
    scale = 1.25

    Xrange_lim = max_x * scale
    Yrange_lim = max_y * scale
    Zrange_lim = max_z * scale
    Wrange_lim = max_w * scale
    '''

    # 投影図の楕円の係数を計算する関数
    # fun4=@(x,y,z) RM(1,1).*x.^2+RM(2,2).*y.^2+RM(3,3).*z.^2+2*RM(1,2).*x.*y+2*RM(1,3).*x.*z+2*RM(2,3).*y.*z-2*log(2);
    def ellipse_coefficients(RM, log2, plane=("x", "z")):
        """
        4次元分解能行列 RM から、指定した2軸 (plane) の断面楕円を求める
        plane: 例 ("x","z"), ("y","w"), ("x","y") など
        """

        # 軸マップ（x=Q//, y=Q⊥, z=E, w=out-of-plane）
        axis_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        i = axis_map[plane[0]]
        j = axis_map[plane[1]]

        # 選んだ2軸以外を消去対象にする
        all_idx = {0, 1, 2, 3}
        elim_idx = list(all_idx - {i, j})

        # 部分行列に分割
        M = RM[np.ix_([i, j], [i, j])]           # 取り出す平面の2x2ブロック
        B = RM[np.ix_([i, j], elim_idx)]         # クロスターン
        C = RM[np.ix_(elim_idx, elim_idx)]       # 消去対象ブロック

        # Schur complement: 有効2D行列
        if C.size > 0:
            C_inv = np.linalg.inv(C)
            M_eff = M - B @ C_inv @ B.T
        else:
            M_eff = M

        # 2D二次形式の係数
        A = M_eff[0, 0]
        Cc = M_eff[1, 1]
        Bc = 2 * M_eff[0, 1]
        D, E = 0, 0
        F = -2 * log2

        return A, Bc, Cc, D, E, F
    
    def ellipse_slice_coefficients(RM, free_axes):
        """
        free_axes: ("x","z") のように残す2軸
        他の軸は0で固定（slice）
        """
        # 軸マップ（x=Q//, y=Q⊥, z=E, w=out-of-plane）
        axes_map = {"x":0, "y":1, "z":2, "w":3}
        
        i, j = axes_map[free_axes[0]], axes_map[free_axes[1]]
        
        A = RM[np.ix_([i,j],[i,j])]
        
        A_xx = A[0,0]
        A_xy = 2*A[0,1]
        A_yy = A[1,1]
        
        return A_xx, A_xy, A_yy, 0, 0, -2*np.log(2)

    # xz平面の楕円の係数
    A_xz, B_xz, C_xz, D_xz, E_xz, F_xz = ellipse_coefficients(RM, log2=np.log(2), plane=("x","z"))

    # yz平面の楕円の係数
    A_yz, B_yz, C_yz, D_yz, E_yz, F_yz = ellipse_coefficients(RM, log2=np.log(2), plane=("y","z"))
    
    # xy平面の楕円の係数
    A_xy, B_xy, C_xy, D_xy, E_xy, F_xy = ellipse_coefficients(RM, log2=np.log(2), plane=("x","y"))
    
    # wz平面の楕円の係数
    A_wz, B_wz, C_wz, D_wz, E_wz, F_wz = ellipse_coefficients(RM, log2=np.log(2), plane=("w","z"))

    # 楕円球の係数行列 RM と楕円球の方程式
    # 4変数対応: x, y, z, w
    def fun4(x, y, z, w, RM):
        return (
            RM[0, 0] * x**2
            + RM[1, 1] * y**2
            + RM[2, 2] * z**2
            + RM[3, 3] * w**2
            + 2 * RM[0, 1] * x * y
            + 2 * RM[0, 2] * x * z
            + 2 * RM[0, 3] * x * w
            + 2 * RM[1, 2] * y * z
            + 2 * RM[1, 3] * y * w
            + 2 * RM[2, 3] * z * w
            - 2 * np.log(2)
        )
    
    # 制約条件（楕円球の式 = 0 を満たす）
    # 制約条件
    def constraint(params, RM):
        x, y, z, w = params
        return fun4(x, y, z, w, RM)
            
    # 最大値を探索する関数
    # 最大値を探索
    def find_max_along_axis(RM, axis="x"):
        initial_guess = [0, 0, 0, 0]  # 4次元原点
        axis_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        idx = axis_map[axis]

        def objective(params):
            return -params[idx]  # 最大化したいので符号反転

        constraints = {"type": "eq", "fun": constraint, "args": (RM,)}

        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            constraints=constraints,
            options={"disp": False},
        )
        return result.x[idx], result.x  # 軸方向の最大値と座標
    
    # 各軸の最大値を計算
    max_x, coords_x = find_max_along_axis(RM, axis="x")# Q//
    max_y, coords_y = find_max_along_axis(RM, axis="y")# Q⊥
    max_z, coords_z = find_max_along_axis(RM, axis="z")# E
    max_w, coords_w = find_max_along_axis(RM, axis="w")# w
    
    # 楕円をプロットする関数
    def plot_ellipse1(A, B, C, D, E, F, Xrange_lim, Zrange_lim, ax, labels, color,ls,shift_x=0,shift_y=0):
        x = np.linspace(-Xrange_lim, Xrange_lim, 50)
        z = np.linspace(-Zrange_lim, Zrange_lim, 50)
        X, Z = np.meshgrid(x, z)

        # 楕円の式
        ellipse = A * X**2 + B * X * Z + C * Z**2 + D * X + E * Z + F
        
        # y方向にhwだけずらす
        X_shifted = X + shift_x
        # y方向にhwだけずらす
        Z_shifted = Z + shift_y
        
        # 表示用 x軸を Qx のノルムで割る
        Qx_norm = np.linalg.norm(Qx)
        X_display = X_shifted / Qx_norm

        # 等高線をプロット（楕円の曲線部分）
        #plt.contour(X_shifted, Z_shifted, ellipse, levels=[0], colors=color, label=label)
        ax.contour(X_display, Z_shifted, ellipse, levels=[0], colors=color, label=labels,linestyles=ls)
    
    def plot_ellipse2(A, B, C, D, E, F, Xrange_lim, Zrange_lim, ax, labels, color,ls,shift_x=0,shift_y=0):
        x = np.linspace(-Xrange_lim, Xrange_lim, 50)
        z = np.linspace(-Zrange_lim, Zrange_lim, 50)
        X, Z = np.meshgrid(x, z)

        # 楕円の式
        ellipse = A * X**2 + B * X * Z + C * Z**2 + D * X + E * Z + F
        
        # y方向にhwだけずらす
        X_shifted = X + shift_x
        # y方向にhwだけずらす
        Z_shifted = Z + shift_y
        
        # 表示用 x軸を Qx のノルムで割る
        Qy_norm = np.linalg.norm(Qy)
        X_display = X_shifted / Qy_norm

        # 等高線をプロット（楕円の曲線部分）
        #plt.contour(X_shifted, Z_shifted, ellipse, levels=[0], colors=color, label=label)
        ax.contour(X_display, Z_shifted, ellipse, levels=[0], colors=color, label=labels,linestyles=ls)
        
    def plot_ellipse3(A, B, C, D, E, F, Xrange_lim, Zrange_lim, ax, labels, color,ls,shift_x=0,shift_y=0):
        x = np.linspace(-Xrange_lim, Xrange_lim, 50)
        z = np.linspace(-Zrange_lim, Zrange_lim, 50)
        X, Z = np.meshgrid(x, z)

        # 楕円の式
        ellipse = A * X**2 + B * X * Z + C * Z**2 + D * X + E * Z + F
        
        # y方向にhwだけずらす
        X_shifted = X + shift_x
        # y方向にhwだけずらす
        Z_shifted = Z + shift_y
        
        # 表示用 x軸を Qx のノルムで割る
        Qx_norm = np.linalg.norm(Qx)
        X_display = X_shifted / Qx_norm
        Qy_norm = np.linalg.norm(Qy)
        Y_display = Z_shifted / Qy_norm

        # 等高線をプロット（楕円の曲線部分）
        #plt.contour(X_shifted, Z_shifted, ellipse, levels=[0], colors=color, label=label)
        ax.contour(X_display, Y_display, ellipse, levels=[0], colors=color, label=labels,linestyles=ls)
        
    def plot_ellipse4(A, B, C, D, E, F, Wrange_lim, Zrange_lim, ax, labels, color,ls,shift_x=0,shift_y=0):
        x = np.linspace(-Wrange_lim, Wrange_lim, 50)
        z = np.linspace(-Zrange_lim, Zrange_lim, 50)
        X, Z = np.meshgrid(x, z)

        # 楕円の式
        ellipse = A * X**2 + B * X * Z + C * Z**2 + D * X + E * Z + F
        
        # y方向にhwだけずらす
        X_shifted = X + shift_x
        # y方向にhwだけずらす
        Z_shifted = Z + shift_y
        
        # 表示用 x軸を Qx のノルムで割る
        Qz_norm = np.linalg.norm(Qz)
        X_display = X_shifted / Qz_norm

        # 等高線をプロット（楕円の曲線部分）
        #plt.contour(X_shifted, Z_shifted, ellipse, levels=[0], colors=color, label=label)
        ax.contour(X_display, Z_shifted, ellipse, levels=[0], colors=color, label=labels,linestyles=ls)
            
    # x=Q//,y=Q⊥,z=E,w=out of plane
    plot_ellipse1(A_xz, B_xz, C_xz, D_xz, E_xz, F_xz, Xrange_lim, Zrange_lim, ax1, labels = "", color="red",ls=["-"],shift_x=0, shift_y=0)
    plot_ellipse2(A_yz, B_yz, C_yz, D_yz, E_yz, F_yz, Yrange_lim, Zrange_lim, ax2, labels = "", color="blue",ls=["-"],shift_x=0, shift_y=0)
    plot_ellipse3(A_xy, B_xy, C_xy, D_xy, E_xy, F_xy, Xrange_lim, Yrange_lim, ax3, labels = "", color="black",ls=["-"],shift_x=0, shift_y=0)
    plot_ellipse4(A_wz, B_wz, C_wz, D_wz, E_wz, F_wz, Wrange_lim, Zrange_lim, ax4, labels = "", color="green",ls=["-"],shift_x=0, shift_y=0)

    A_xz_s, B_xz_s, C_xz_s, D_xz_s, E_xz_s, F_xz_s = ellipse_slice_coefficients(RM, ("x","z"))
    plot_ellipse1(A_xz_s, B_xz_s, C_xz_s, D_xz_s, E_xz_s, F_xz_s,
            Xrange_lim, Zrange_lim, ax1,
            labels = "",color="red",ls=["--"], 
            shift_x=0, shift_y=0)
    A_yz_s, B_yz_s, C_yz_s, D_yz_s, E_yz_s, F_yz_s = ellipse_slice_coefficients(RM, ("y","z"))
    plot_ellipse2(A_yz_s, B_yz_s, C_yz_s, D_yz_s, E_yz_s, F_yz_s,
            Yrange_lim, Zrange_lim, ax2,
            labels = "",color="blue",ls=["--"],
            shift_x=0, shift_y=0)
    A_xy_s, B_xy_s, C_xy_s, D_xy_s, E_xy_s, F_xy_s = ellipse_slice_coefficients(RM, ("x","y"))
    plot_ellipse3(A_xy_s, B_xy_s, C_xy_s, D_xy_s, E_xy_s, F_xy_s,
            Xrange_lim, Yrange_lim, ax3,
            labels = "",color="black",ls=["--"],
            shift_x=0, shift_y=0)
    A_wz_s, B_wz_s, C_wz_s, D_wz_s, E_wz_s, F_wz_s = ellipse_slice_coefficients(RM, ("w","z"))
    plot_ellipse4(A_wz_s, B_wz_s, C_wz_s, D_wz_s, E_wz_s, F_wz_s,
            Wrange_lim, Zrange_lim, ax4,
            labels = "",color="green",ls=["--"],
            shift_x=0, shift_y=0)
    
    # 各軸の最大値を2倍した値
    resolution_Q_parallel = 2 * max_x
    resolution_Q_perpendicular = 2 * max_y
    resolution_energy = 2 * max_z
    resolution_Q_z = 2 * max_w
    
    plt.suptitle(
        f'ℏω: {QE_sets[0]} meV, h: {QE_sets[1]}, k: {QE_sets[2]}, l: {QE_sets[3]}\n'
        r'$\delta Q_{x} (\parallel axis1)$ = ' + f'{resolution_Q_parallel/np.linalg.norm(Qx):.4f}' + r' (r.l.u.), '
        r'$\delta Q_{y} (\parallel axis2)$ = ' + f'{resolution_Q_perpendicular/np.linalg.norm(Qy):.4f}' + r' (r.l.u.), '
        r'$\delta Q_{z} (\parallel axis3)$ = ' + f'{resolution_Q_z/np.linalg.norm(Qz):.4f}' + r' (r.l.u.), '
        f'δℏω = {resolution_energy:.4f}'  + r' (meV)',
        fontsize=11,
        y=0.98  # 上の余白を調整したい場合に使用（デフォルトより少し上）
    )
    
    # === Q_parallel vs E の楕円描画 ===
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax1.set_xlabel(r"$\delta Q_{x}$ (r.l.u.)")
    ax1.set_ylabel("δℏω (meV)")
    ax1.set_title(r"$Q_{x} \parallel$" + f"({sv1[0]:.4f}, {sv1[1]:.4f}, {sv1[2]:.4f})", fontsize=12)

    ax1.set_xlim([-Xrange_lim/np.linalg.norm(Qx), Xrange_lim/np.linalg.norm(Qx)])
    ax1.set_ylim([-Zrange_lim, Zrange_lim])
    ax1.grid(True)

    # === Q_perp vs E の楕円描画===
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xlabel(r"$\delta Q_{y}$ (r.l.u.)")
    ax2.set_ylabel("δℏω (meV)")
    ax2.set_title(r"$Q_{y} \parallel$" + f"({sv2[0]:.4f}, {sv2[1]:.4f}, {sv2[2]:.4f})", fontsize=12)

    # 必要であれば同様に情報を追加（または省略）
    ax2.set_xlim([-Yrange_lim/np.linalg.norm(Qy), Yrange_lim/np.linalg.norm(Qy)])
    ax2.set_ylim([-Zrange_lim, Zrange_lim])
    ax2.grid(True)
    
    # === Q_perp vs Q_parallelの楕円描画===
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax3.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax3.set_xlabel(r"$\delta Q_{x}$ (r.l.u.)")
    ax3.set_ylabel(r"$\delta Q_{y}$ (r.l.u.)")
    ax3.set_title(r"$\delta Q_{x} ({\parallel}axis1)$ vs $\delta Q_{y} ({\parallel}axis2)$ ellipse", fontsize=12)

    # 必要であれば同様に情報を追加（または省略）
    ax3.set_xlim([-Xrange_lim/np.linalg.norm(Qx), Xrange_lim/np.linalg.norm(Qx)])
    ax3.set_ylim([-Yrange_lim/np.linalg.norm(Qy), Yrange_lim/np.linalg.norm(Qy)])
    #ax3.set_aspect(np.linalg.norm(Qy)/np.linalg.norm(Qx))  # ここで縦横比を1:1に固定
    ax3.grid(True)
    
    # === Q_perp vs E の楕円描画===
    ax4.axhline(0, color="green", linestyle="--", linewidth=0.5)
    ax4.axvline(0, color="green", linestyle="--", linewidth=0.5)
    ax4.set_xlabel(r"$\delta Q_{z}$ (r.l.u.)")
    ax4.set_ylabel("δℏω (meV)")
    ax4.set_title(r"$Q_{z} \parallel$" + f"({sv3[0]:.4f}, {sv3[1]:.4f}, {sv3[2]:.4f})", fontsize=12)

    # 必要であれば同様に情報を追加（または省略）
    ax4.set_xlim([-Wrange_lim/np.linalg.norm(Qz), Wrange_lim/np.linalg.norm(Qz)])
    ax4.set_ylim([-Zrange_lim, Zrange_lim])
    ax4.grid(True)

    #return A_sets,QE_sets,RM, fig
    return fig
