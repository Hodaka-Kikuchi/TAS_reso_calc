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

def calcresolution_scan3(lc_param,rl,col_param,mos_param,config,approximation,focusing,geom,calc_params,unit_mode):

    # divergenceの読み出し
    gm_1st = col_param["gm_1st"]
    if not gm_1st:
        div_1st_h = col_param["div_1st_h"]
        div_1st_v = col_param["div_1st_v"]
    else:
        div_1st_m = col_param["div_1st_m"]
    div_2nd_h = col_param["div_2nd_h"]
    div_2nd_v = col_param["div_2nd_v"]
    div_3rd_h = col_param["div_3rd_h"]
    div_3rd_v = col_param["div_3rd_v"]
    div_4th_h = col_param["div_4th_h"]
    div_4th_v = col_param["div_4th_v"]

    # focusing conditionの読み出し
    # Monochromator Horizontal
    MHF = focusing["monochromator"]["horizontal"]["enabled"]
    num_mono_h = focusing["monochromator"]["horizontal"]["blades"]

    # Monochromator Vertical
    MVF = focusing["monochromator"]["vertical"]["enabled"]
    num_mono_v = focusing["monochromator"]["vertical"]["blades"]

    # Analyzer Horizontal
    AHF = focusing["analyzer"]["horizontal"]["enabled"]
    num_ana_h = focusing["analyzer"]["horizontal"]["blades"]

    # Analyzer Vertical
    AVF = focusing["analyzer"]["vertical"]["enabled"]
    num_ana_v = focusing["analyzer"]["vertical"]["blades"]

    # geom
    L0 = geom["L0"]
    L1 = geom["L1"]
    L2 = geom["L2"]
    L3 = geom["L3"]
    beam_width = geom["beam_width"]
    beam_height = geom["beam_height"]
    mono_width = geom["mono_width"]
    mono_height = geom["mono_height"]
    mono_depth = geom["mono_thickness"]
    ana_width = geom["ana_width"]
    ana_height = geom["ana_height"]
    ana_depth = geom["ana_thickness"]
    det_width = geom["det_width"]
    det_height = geom["det_height"]

    #
    d_mono = mos_param["d_mono"]
    mos_mono_h = mos_param["mos_mono_h"]
    mos_mono_v = mos_param["mos_mono_v"]
    mos_sam_h = mos_param["mos_sam_h"]
    mos_sam_v = mos_param["mos_sam_v"]
    d_ana = mos_param["d_ana"]
    mos_ana_h = mos_param["mos_ana_h"]
    mos_ana_v = mos_param["mos_ana_v"]

    energy_mode = config["energy_mode"]
    if energy_mode == "Ei fixed":
        Ei = config["Ei"]
    else:
        Ef = config["Ef"]
    geometry = config["geometry"]
    sense = config["sign_config"]
    method = approximation["method"]

    #
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

    ####################
    if energy_mode == "Ei fixed":
        Ef = Ei - cphw
    else:
        Ei = Ef + cphw
    C1 = np.degrees(np.arcsin((2 * np.pi / d_mono) / (2 * np.sqrt(Ei / 2.072))))
    C3 = np.degrees(np.arcsin((2 * np.pi / d_ana) / (2 * np.sqrt(Ef / 2.072))))
    if geometry == "anti-W":
        C3 = -C3
    ki_cal=(Ei/2.072)**(1/2)
    kf_cal=(Ef/2.072)**(1/2)
    hkl_cal=cph*astar+cpk*bstar+cpl*cstar
    Nhkl_cal=np.linalg.norm(hkl_cal)
    phi_cal = np.degrees(np.arccos((ki_cal**2 + kf_cal**2 - Nhkl_cal**2) / (2 * ki_cal * kf_cal)))
    A_sets = [2*C1, phi_cal, 2*C3]
    QE_sets = [cphw, cph, cpk, cpl]
    ####################
    
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
    
    # ここでscanの最初と最後のポイントの分解能の計算する。
    # 空リストに格納
    X_vals, Y_vals, Z_vals, W_vals = [], [], [], []
    
    A1, A2, A3 = A_sets
    A2 = -A2
    hw = QE_sets[0]
    
    if energy_mode == "Ei fixed":
        Ef = Ei - hw
    else:
        Ei = Ef + hw
    
    ki=(Ei/2.072)**(1/2)
    kf=(Ef/2.072)**(1/2)

    Q = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(np.radians(A2))) 

    # C1とA1の計算
    C1 = A1/2

    # C3とA3の計算
    C3 = A3/2

    thetaM = C1
    thetaS = A2 / 2
    thetaA = C3
    
    phi = np.degrees(np.arctan2(-kf * np.sin(np.radians(2 * thetaS)), ki - kf * np.cos(np.radians(2 * thetaS))))

    # Define constants for the resolution matrices
    # ここでαi とβi は、コリメータの水平方向と鉛直方向における発散角を表している。η とη′をモノクロメータとアナライザの水平方向と鉛直方向のモザイクのFWHM

    #theta0 = 0.1 #(A^-1)
    #lamda = (81.81 / Ei)**(1/2)
    # 0.4246609 = 1/(2*sqrt(2*log(2)))
    if not gm_1st:
        alpha1 = div_1st_h / 60 / 180 * pi * 0.4246609
        beta1 = div_1st_v / 60 / 180 * pi * 0.4246609 
    else:
        NA = 6.022*10**(23) # mol^(-1)
        ro = 8.908 # g/cm^2
        M = 58.69 # g/mol
        bc = 1.03*10**(-12) # cm
        lamda=sqrt(81.81/Ei)*10**(-8) # cm
        Qc = 0.0219 # = sqrt(16*pi*po)
        div_1st_h = div_1st_m*2*np.degrees(arcsin(lamda*sqrt(NA*ro/M*bc/pi)))*60
        div_1st_v = div_1st_h
        alpha1 = div_1st_m*2*np.degrees(arcsin(lamda*sqrt(NA*ro/M*bc/pi)))*60  / 60 / 180 * pi * 0.4246609 
        beta1 = alpha1
        
    alpha2 = div_2nd_h / 60 / 180 * pi * 0.4246609
    # focusingの場合式が異なる。
    
    if AHF:
        L=L2
        W=ana_width*num_ana_h*np.sin(np.radians(A3))
        af=2 * np.degrees(np.arctan((W / 2) / L))
        #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
        alpha3 = af / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
        
        #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
    else:
        alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609
        
    alpha4 = div_4th_h / 60 / 180 * pi * 0.4246609
    beta2 = div_2nd_v / 60 / 180 * pi * 0.4246609
    beta3 = div_3rd_v / 60 / 180 * pi * 0.4246609
    beta4 = div_4th_v / 60 / 180 * pi * 0.4246609
    
    etaM = mos_mono_h / 60 / 180 * pi * 0.4246609
    etaA = mos_ana_h / 60 / 180 * pi * 0.4246609
    etaS = mos_sam_h / 60 / 180 * pi * 0.4246609
    etaMp = mos_mono_v / 60 / 180 * pi * 0.4246609
    etaAp = mos_ana_v / 60 / 180 * pi * 0.4246609
    etaSp = mos_sam_v / 60 / 180 * pi * 0.4246609

    # Gについてreslibと同じ値になることを確認
    G = 8 * np.log(2) / (8 * np.log(2)) * np.diag([1 / (alpha1 ** 2), 1 / (alpha2 ** 2), 1 / (beta1 ** 2), 1 / (beta2 ** 2), 1 / (alpha3 ** 2), 1 / (alpha4 ** 2), 1 / (beta3 ** 2), 1 / (beta4 ** 2)])
    # Fについてreslibと同じ値になることを確認
    F = 8 * np.log(2) / (8 * np.log(2))  * np.diag([1 / (etaM ** 2), 1 / (etaMp ** 2), 1 / (etaA ** 2), 1 / (etaAp ** 2)])
    
    # Define matrices A, B, and C
    A = np.zeros((6, 8))# reslibと一致することを確認
    C = np.zeros((4, 8))# reslibと一致することを確認
    B = np.zeros((4, 6))# reslibと一致することを確認
    
    A[0, 0] = ki / (2 * np.tan(np.radians(thetaM)))
    A[0, 1] = -A[0, 0]
    A[3, 4] = kf / (2 * np.tan(np.radians(thetaA)))
    A[3, 5] = -A[3, 4]
    A[1, 1] = ki
    A[2, 3] = ki
    A[4, 4] = kf
    A[5, 6] = kf
    
    # 2.072142=h^2/m
    B[0, 0] = np.cos(np.radians(phi))
    B[0, 1] = np.sin(np.radians(phi))
    B[0, 3] = -np.cos(np.radians(phi - 2 * thetaS))
    B[0, 4] = -np.sin(np.radians(phi - 2 * thetaS))
    B[1, 0] = -B[0, 1]
    B[1, 1] = B[0, 0]
    B[1, 3] = -B[0, 4]
    B[1, 4] = B[0, 3]
    B[2, 2] = 1
    B[2, 5] = -1
    B[3, 0] = 2 * 2.072142 * ki
    B[3, 3] = -2 * 2.072142 * kf

    C[0, 0] = 1 / 2
    C[0, 1] = 1 / 2
    C[2, 4] = 1 / 2
    C[2, 5] = 1 / 2
    C[1, 2] = 1 / (2 * sin(np.radians(thetaM)))
    C[1, 3] = -C[1, 2]
    C[3, 6] = 1 / (2 * sin(np.radians(thetaA)))
    C[3, 7] = -C[3, 6]
    
    # popovic近似へ分岐
    if method == "Popovici":
        # ==== Beam shape ====
        beamw = (beam_width)**2
        beamh = (beam_height)**2
        bshape = np.diag([beamw, beamh])

        # ==== Mono shape ====
        monow = (num_mono_h*mono_width)**2
        monoh = (num_mono_v*mono_height)**2
        monod = (mono_depth)**2
        mshape = np.diag([monod, monow, monoh])

        '''
        # ==== Monitor shape ====
        # only flux normalization
        monitorw = 1
        monitorh = 1
        # if you want:
        #monitorw = monitor.width**2
        #monitorh = monitor.height**2
        monitorshape = np.diag([monitorw, monitorh])
        '''

        # ==== Sample shape ====
        sshape = np.eye(3)
        psi = thetaS -phi
        rot = np.array([[cos(np.radians(psi)),sin(np.radians(psi)),0],
            [-sin(np.radians(psi)),cos(np.radians(psi)),0],
            [0,0,1]])
        sshape = rot@sshape@rot.T

        # ==== ana shape ====
        anaw = (num_ana_h*ana_width)**2
        anah = (num_ana_v*ana_height)**2
        anad = (ana_depth)**2
        ashape = np.diag([anad, anaw, anah])

        # ==== det shape ====
        detw = (det_width)**2
        deth = (det_height)**2
        dshape = np.diag([detw, deth])

        # ==== S matrix ====
        Sinv = block_diag(bshape, mshape, sshape, ashape, dshape)  # S^-1
        S = np.linalg.inv(Sinv)

        # ==== Distances ====

        def Vfocusing_curvature(L_1, L_2, theta):
            # 有効焦点距離。ただし、単位をmmからmに直す必要がある。
            f = 1.0 / (1.0/L_1 + 1.0/L_2)

            # θ をラジアンに変換
            theta_rad = np.radians(theta)

            # 曲率 R = 2*f*|sin(theta)|
            R = 2.0 * f * np.abs(np.sin(theta_rad))

            return R
        
        def Hfocusing_curvature(L_1, L_2, theta):
            # 有効焦点距離。ただし、単位をmmからmに直す必要がある。
            f = 1.0 / (1.0/L_1 + 1.0/L_2)

            # θ をラジアンに変換
            theta_rad = np.radians(theta)

            # 曲率 R = 2*f*|sin(theta)|
            R = 2.0 * f / np.abs(np.sin(theta_rad))

            return R
        
        # focusing:
        if MVF:
            monorv = Vfocusing_curvature(L0,L1,thetaM)
        else:
            monorv = 1e10
        if MHF:
            monorh = Hfocusing_curvature(L0,L1,thetaM)
        else:
            monorh = 1e10
        if AVF:
            anarv = Vfocusing_curvature(L2,L3,thetaA)
        else:
            anarv = 1e10
        if AHF:
            anarh = Hfocusing_curvature(L2,L3,thetaA)
        else:
            anarh = 1e10

        # ==== T matrix ====
        T = np.zeros((4, 13))

        T[0, 0] = -1/(2*L0)
        T[0, 2] = np.cos(np.radians(thetaM))*(1/L1 - 1/L0)/2
        T[0, 3] = np.sin(np.radians(thetaM))*(1/L0 + 1/L1 - 2/(monorh*np.sin(np.radians(thetaM))))/2
        T[0, 5] = np.sin(np.radians(thetaS))/(2*L1)
        T[0, 6] = np.cos(np.radians(thetaS))/(2*L1)

        T[1, 1] = -1/(2*L0*np.sin(np.radians(thetaM)))
        T[1, 4] = (1/L0 + 1/L1 - 2*np.sin(np.radians(thetaM))/monorv)/(2*np.sin(np.radians(thetaM)))
        T[1, 7] = -1/(2*L1*np.sin(np.radians(thetaM)))

        T[2, 5] = np.sin(np.radians(thetaS))/(2*L2)
        T[2, 6] = -np.cos(np.radians(thetaS))/(2*L2)
        T[2, 8] = np.cos(np.radians(thetaA))*(1/L3 - 1/L2)/2
        T[2, 9] = np.sin(np.radians(thetaA))*(1/L2 + 1/L3 - 2/(anarh*np.sin(np.radians(thetaA))))/2
        T[2, 11] = 1/(2*L3)

        T[3, 7] = -1/(2*L2*np.sin(np.radians(thetaA)))
        T[3, 10] = (1/L2 + 1/L3 - 2*np.sin(np.radians(thetaA))/anarv)/(2*np.sin(np.radians(thetaA)))
        T[3, 12] = -1/(2*L3*np.sin(np.radians(thetaA)))

        # ==== D matrix ====
        D = np.zeros((8, 13))

        D[0, 0] = -1/L0
        D[0, 2] = -np.cos(np.radians(thetaM))/L0
        D[0, 3] = np.sin(np.radians(thetaM))/L0

        D[2, 1] = D[0, 0]
        D[2, 4] = -D[0, 0]

        D[1, 2] = np.cos(np.radians(thetaM))/L1
        D[1, 3] = np.sin(np.radians(thetaM))/L1
        D[1, 5] = np.sin(np.radians(thetaS))/L1
        D[1, 6] = np.cos(np.radians(thetaS))/L1

        D[3, 4] = -1/L1
        D[3, 7] = -D[3, 4]

        D[4, 5] = np.sin(np.radians(thetaS))/L2
        D[4, 6] = -np.cos(np.radians(thetaS))/L2
        D[4, 8] = -np.cos(np.radians(thetaA))/L2
        D[4, 9] = np.sin(np.radians(thetaA))/L2

        D[6, 7] = -1/L2
        D[6, 10] = -D[6, 7]

        D[5, 8] = np.cos(np.radians(thetaA))/L3
        D[5, 9] = np.sin(np.radians(thetaA))/L3
        D[5, 11] = 1/L3

        D[7, 10] = -D[5, 11]
        D[7, 12] = D[5, 11]

    # 計算
    term = np.linalg.inv(G + C.T @ F @ C)  # G + C' * F * C の逆行列
    HF = A @ term @ A.T  # A * (G + C' * F * C)^(-1) * A'
    # HFまでreslibと一致
    if method == "Popovici":
        Minv = (B @ A @ np.linalg.inv(np.linalg.inv(D @ np.linalg.inv(S + T.T @ F @ T) @ D.T) + G) @ A.T @ B.T)
    elif method == "Cooper-Nathans":
        if AHF:
            P = np.linalg.inv(HF)
            P[4, 4] = (1 / (kf * alpha3)) ** 2
            P[3, 4] = 0
            P[3, 3] = (np.tan(np.radians(thetaA)) / (etaA * kf)) ** 2
            P[4, 3] = 0
            Pinv = np.linalg.inv(P)
            Minv = B @ Pinv @ B.T
        else:
            Minv = B @ HF @ B.T #これもreslibと一致
            
    M = np.linalg.inv(Minv)
    
    # RM 行列の設定
    #RM = np.zeros((4, 4))  # 4x4 のゼロ行列で初期化
    """
    RM = [[M[0, 0],M[0, 1],M[0, 3],M[0, 2]],
        [M[1, 0],M[1, 1],M[1, 3],M[1, 2]],
        [M[3, 0],M[3, 1],M[3, 3],M[3, 2]],
        [M[2, 0],M[2, 1],M[2, 3],M[2, 2]]]
    """
    # 軸 2↔3 をスワップするインデックス
    swap = [0, 1, 3, 2]
    RM = M[np.ix_(swap, swap)]
    
    # サンプルモザイクを入れた場合の計算
    Minv = np.linalg.inv(RM)
    Minv[1, 1] += Q**2 * etaS**2# / (8 * np.log(2))
    Minv[3, 3] += Q**2 * etaSp**2# / (8 * np.log(2))
    RM = np.linalg.inv(Minv)
    
    # ============================================================
    # 座標変換
    # ============================================================

    # 結晶の U, V, W ベクトル
    Qx = sv1[0]*astar + sv1[1]*bstar + sv1[2]*cstar   # U
    Qy = sv2[0]*astar + sv2[1]*bstar + sv2[2]*cstar   # V
    Qz = sv3[0]*astar + sv3[1]*bstar + sv3[2]*cstar   # W

    # 測定しているQベクトル
    Qvect = QE_sets[1]*astar + QE_sets[2]*bstar + QE_sets[3]*cstar

    # RLU用のnormを計算
    if unit_mode == "$\mathrm{\AA}^{-1}$":
        normQx = 1.0
        normQy = 1.0
        normQz = 1.0

    elif unit_mode == "(r.l.u.)":
        normQx = np.linalg.norm(Qx)
        normQy = np.linalg.norm(Qy)
        normQz = np.linalg.norm(Qz)

    # ============================================================
    # 単位ベクトル
    # ============================================================

    uq = Qvect / np.linalg.norm(Qvect)
    u_hat = Qx / np.linalg.norm(Qx)
    v_hat = Qy / np.linalg.norm(Qy)
    w_hat = Qz / np.linalg.norm(Qz)

    uv_angle = np.arctan2(
        np.dot(np.cross(u_hat, v_hat), w_hat),
        np.dot(u_hat, v_hat)
    )
    

    # ============================================================
    # 現在のRMの直交座標系
    #
    # x : Q方向
    # y : 散乱面内Q⊥方向
    # z : 散乱面外方向
    #
    # QはU-V散乱面内にあるため、
    # VをQに垂直な方向へ射影してy軸を作る
    # ============================================================

    # ============================================================
    # Q方向を新しい x 軸とする
    # ============================================================

    e_x = uq

    # ============================================================
    # Qに垂直な面内方向を V から作る
    # ============================================================

    V_perp = v_hat - np.dot(v_hat, e_x) * e_x

    if np.linalg.norm(V_perp) > 1e-12:

        e_y = V_perp / np.linalg.norm(V_perp)

    else:

        # V || Q の場合は U を使う
        U_perp = u_hat - np.dot(u_hat, e_x) * e_x

        if np.linalg.norm(U_perp) > 1e-12:
            e_y = U_perp / np.linalg.norm(U_perp)
        else:
            raise ValueError(
                "U, V ともに Q と平行で e_y を定義できません。"
            )

    # ============================================================
    # 右手系を維持
    # ============================================================

    e_z = np.cross(e_x, e_y)
    e_z /= np.linalg.norm(e_z)

    # ============================================================
    # 結晶軸をRMの直交座標系に射影して角度を求める
    #
    # 元コードと同じ定義：
    #
    # theta = atan2(y, x)
    # tilt  = atan2(z, sqrt(x^2+y^2))
    #
    # ここで x,y,z は「RM座標系から見た結晶軸」の成分
    # ============================================================

    def get_theta_tilt(axis):

        axis_hat = axis / np.linalg.norm(axis)

        x = np.dot(axis_hat, e_x)
        y = np.dot(axis_hat, e_y)
        z = np.dot(axis_hat, e_z)

        theta = -np.arctan2(y, x)

        q_uv = np.sqrt(x**2 + y**2)
        tilt = -np.arctan2(z, q_uv)

        return theta, tilt


    # ============================================================
    # U, V, W の角度
    # ============================================================

    theta_U, tilt_U = get_theta_tilt(Qx)
    theta_V, tilt_V = get_theta_tilt(Qy)
    theta_W, tilt_W = get_theta_tilt(Qz)

    #print("=== UVW rotation angles ===")
    #print(f"U: theta = {np.degrees(theta_U):.6f} deg, tilt = {np.degrees(tilt_U):.6f} deg")
    #print(f"V: theta = {np.degrees(theta_V):.6f} deg, tilt = {np.degrees(tilt_V):.6f} deg")
    #print(f"W: theta = {np.degrees(theta_W):.6f} deg, tilt = {np.degrees(tilt_W):.6f} deg")
    #print("============================")

    # ============================================================
    # TASではU,Vは散乱面内
    # Wだけ散乱面外のtiltを持つ
    # ============================================================

    tilt_U = 0.0
    tilt_V = 0.0

    # ============================================================
    # 回転行列
    # ============================================================

    def make_rot_mat(theta_rad, tilt_rad=0.0):

        # 散乱面内の回転
        R_theta = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
            [np.sin(theta_rad),  np.cos(theta_rad), 0, 0],
            [0,                  0,                  1, 0],
            [0,                  0,                  0, 1]
        ])

        # 散乱面外の回転
        R_tilt = np.array([
            [np.cos(tilt_rad), 0, 0, -np.sin(tilt_rad)],
            [0,                1, 0,  0],
            [0,                0, 1,  0],
            [np.sin(tilt_rad), 0, 0,  np.cos(tilt_rad)]
        ])

        return R_tilt @ R_theta


    # ============================================================
    # U方向の分解能
    # ============================================================

    rot_U = make_rot_mat(theta_U, tilt_U)
    RM_U = rot_U @ RM @ rot_U.T


    # ============================================================
    # V方向の分解能
    # ============================================================

    rot_V = make_rot_mat(theta_V, tilt_V)
    RM_V = rot_V @ RM @ rot_V.T

    # ============================================================
    # W方向の分解能
    # ============================================================

    rot_W = make_rot_mat(theta_W, tilt_W)
    RM_W = rot_W @ RM @ rot_W.T

    
    # sense による V/W 方向の反転
    if sense == '-+-':

        S_VW = np.diag([-1.0, 1.0, 1.0, -1.0])

        RM_V = S_VW @ RM_V @ S_VW.T
        RM_W = S_VW @ RM_W @ S_VW.T

    elif sense == '+-+':
        pass
    
    
    # RMは(q//,q⊥,hw,qz)における空間分布
    # これを(qx(axis1),qy(axis2),hw,qz)に置ける空間分布に変換する。
    
    # プロット範囲
    #Xrange_lim = 0.1
    #Zrange_lim = 0.5
    
    # Qx=Q//,Qy=Q⊥の定義
    
    # ============================================================
    # 楕円球の係数行列 RM と楕円球の方程式
    #
    # x = Q方向
    # y = Q方向に垂直な散乱面内方向
    # z = Energy
    # w = 散乱面外方向
    # ============================================================

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


    # ============================================================
    # 制約条件
    # ============================================================

    def constraint(params, RM):

        x, y, z, w = params

        return fun4(x, y, z, w, RM)


    # ============================================================
    # 各軸方向の最大値を探索
    # ============================================================

    def find_max_along_axis(RM, axis="x"):

        initial_guess = [0, 0, 0, 0]

        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            "w": 3
        }

        idx = axis_map[axis]

        def objective(params):

            return -params[idx]

        constraints = {
            "type": "eq",
            "fun": constraint,
            "args": (RM,)
        }

        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            constraints=constraints,
            options={"disp": False},
        )

        return result.x[idx], result.x


    # ============================================================
    # U, V, W, Energy の最大値
    # ============================================================

    # U方向
    max_U, coords_U = find_max_along_axis(
        RM_U,
        axis="x"
    )

    # V方向
    max_V, coords_V = find_max_along_axis(
        RM_V,
        axis="x"
    )

    # W方向
    max_W, coords_W = find_max_along_axis(
        RM_W,
        axis="x"
    )

    # Energy
    max_E, coords_E = find_max_along_axis(
        RM_U,
        axis="z"
    )


    # ============================================================
    # プロット範囲
    # ============================================================

    scale = 1.20

    Urange_lim = max_U * scale
    Vrange_lim = max_V * scale
    Wrange_lim = max_W * scale
    Zrange_lim = max_E * scale


    # ============================================================
    # 投影図の楕円係数
    #
    # RM_U:
    #     x = U
    #     z = E
    #
    # RM_V:
    #     x = V
    #     z = E
    #
    # RM_W:
    #     x = W
    #     z = E
    #
    # したがって全て ("x","z") を使う。
    # ============================================================

    def ellipse_coefficients(RM, log2, plane=("x", "z")):

        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            "w": 3
        }

        i = axis_map[plane[0]]
        j = axis_map[plane[1]]

        # 選択した2軸以外を消去
        all_idx = {0, 1, 2, 3}

        elim_idx = list(all_idx - {i, j})

        # 2x2 block
        M = RM[np.ix_([i, j], [i, j])]

        # Cross term
        B = RM[np.ix_([i, j], elim_idx)]

        # 消去対象
        C = RM[np.ix_(elim_idx, elim_idx)]

        # Schur complement
        if C.size > 0:

            C_inv = np.linalg.inv(C)

            M_eff = M - B @ C_inv @ B.T

        else:

            M_eff = M

        # 2次形式
        A = M_eff[0, 0]
        Cc = M_eff[1, 1]
        Bc = 2 * M_eff[0, 1]

        D = 0
        E = 0

        F = -2 * log2

        return A, Bc, Cc, D, E, F


    # ============================================================
    # Slice の楕円係数
    #
    # x,z 以外を 0 に固定
    # ============================================================

    def ellipse_slice_coefficients(RM, free_axes):

        axes_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            "w": 3
        }

        i = axes_map[free_axes[0]]
        j = axes_map[free_axes[1]]

        A = RM[np.ix_([i, j], [i, j])]

        A_xx = A[0, 0]
        A_xy = 2 * A[0, 1]
        A_yy = A[1, 1]

        return (
            A_xx,
            A_xy,
            A_yy,
            0,
            0,
            -2 * np.log(2)
        )


    # ============================================================
    # U-E 投影楕円
    # ============================================================

    A_U, B_U, C_U, D_U, E_U, F_U = \
        ellipse_coefficients(
            RM_U,
            log2=np.log(2),
            plane=("x", "z")
        )


    # ============================================================
    # V-E 投影楕円
    # ============================================================

    A_V, B_V, C_V, D_V, E_V, F_V = \
        ellipse_coefficients(
            RM_V,
            log2=np.log(2),
            plane=("x", "z")
        )


    # ============================================================
    # W-E 投影楕円
    # ============================================================

    A_W, B_W, C_W, D_W, E_W, F_W = \
        ellipse_coefficients(
            RM_W,
            log2=np.log(2),
            plane=("x", "z")
        )


    # ============================================================
    # U-E slice
    # ============================================================

    A_U_s, B_U_s, C_U_s, D_U_s, E_U_s, F_U_s = \
        ellipse_slice_coefficients(
            RM_U,
            ("x", "z")
        )


    # ============================================================
    # V-E slice
    # ============================================================

    A_V_s, B_V_s, C_V_s, D_V_s, E_V_s, F_V_s = \
        ellipse_slice_coefficients(
            RM_V,
            ("x", "z")
        )


    # ============================================================
    # W-E slice
    # ============================================================

    A_W_s, B_W_s, C_W_s, D_W_s, E_W_s, F_W_s = \
        ellipse_slice_coefficients(
            RM_W,
            ("x", "z")
        )


    # ============================================================
    # 楕円をプロットする関数
    # ============================================================

    def plot_ellipse(
        A, B, C, D, E, F,
        Xrange_lim,
        Zrange_lim,
        ax,
        labels,
        color,
        ls,
        normQ,
        shift_x=0,
        shift_y=0
    ):

        x = np.linspace(
            -Xrange_lim,
            Xrange_lim,
            50
        )

        z = np.linspace(
            -Zrange_lim,
            Zrange_lim,
            50
        )

        X, Z = np.meshgrid(x, z)

        ellipse = (
            A * X**2
            + B * X * Z
            + C * Z**2
            + D * X
            + E * Z
            + F
        )

        X_shifted = X + shift_x
        Z_shifted = Z + shift_y

        # Å^-1 → 表示単位
        X_display = X_shifted / normQ

        ax.contour(
            X_display,
            Z_shifted,
            ellipse,
            levels=[0],
            colors=color,
            label=labels,
            linestyles=ls
        )


    # ============================================================
    # 投影楕円を描画
    # ============================================================

    # U-E
    plot_ellipse(
        A_U,
        B_U,
        C_U,
        D_U,
        E_U,
        F_U,
        Urange_lim,
        Zrange_lim,
        ax1,
        labels="",
        color="red",
        ls=["-"],
        normQ=normQx,
        shift_x=0,
        shift_y=0
    )


    # V-E
    plot_ellipse(
        A_V,
        B_V,
        C_V,
        D_V,
        E_V,
        F_V,
        Vrange_lim,
        Zrange_lim,
        ax2,
        labels="",
        color="blue",
        ls=["-"],
        normQ=normQy,
        shift_x=0,
        shift_y=0
    )


    # W-E
    plot_ellipse(
        A_W,
        B_W,
        C_W,
        D_W,
        E_W,
        F_W,
        Wrange_lim,
        Zrange_lim,
        ax4,
        labels="",
        color="green",
        ls=["-"],
        normQ=normQz,
        shift_x=0,
        shift_y=0
    )


    # ============================================================
    # Slice 楕円を描画
    # ============================================================

    # U-E slice
    plot_ellipse(
        A_U_s,
        B_U_s,
        C_U_s,
        D_U_s,
        E_U_s,
        F_U_s,
        Urange_lim,
        Zrange_lim,
        ax1,
        labels="",
        color="red",
        ls=["--"],
        normQ=normQx,
        shift_x=0,
        shift_y=0
    )


    # V-E slice
    plot_ellipse(
        A_V_s,
        B_V_s,
        C_V_s,
        D_V_s,
        E_V_s,
        F_V_s,
        Vrange_lim,
        Zrange_lim,
        ax2,
        labels="",
        color="blue",
        ls=["--"],
        normQ=normQy,
        shift_x=0,
        shift_y=0
    )


    # W-E slice
    plot_ellipse(
        A_W_s,
        B_W_s,
        C_W_s,
        D_W_s,
        E_W_s,
        F_W_s,
        Wrange_lim,
        Zrange_lim,
        ax4,
        labels="",
        color="green",
        ls=["--"],
        normQ=normQz,
        shift_x=0,
        shift_y=0
    )

    # ============================================================
    # Resolution
    # ============================================================

    resolution_U = 2 * max_U
    resolution_V = 2 * max_V
    resolution_W = 2 * max_W
    resolution_energy = 2 * max_E


    # ============================================================
    # Figure タイトル
    # ============================================================

    # ============================================================
    # Figure タイトル用の表示単位変換
    # ============================================================

    if unit_mode == "$\mathrm{\AA}^{-1}$":

        q_unit = r"$(\mathrm{\AA}^{-1})$"

        resolution_U_display = resolution_U
        resolution_V_display = resolution_V
        resolution_W_display = resolution_W

    elif unit_mode == "(r.l.u.)":

        q_unit = r"$(\mathrm{r.l.u.})$"

        resolution_U_display = resolution_U / normQx
        resolution_V_display = resolution_V / normQy
        resolution_W_display = resolution_W / normQz


    # ============================================================
    # Figure タイトル
    # ============================================================

    plt.suptitle(
        f'ℏω: {QE_sets[0]} meV, '
        f'h: {QE_sets[1]}, '
        f'k: {QE_sets[2]}, '
        f'l: {QE_sets[3]}\n'

        r'$\delta Q_U$ = '
        + f'{resolution_U_display:.4f} '
        + q_unit
        + ', '

        r'$\delta Q_V$ = '
        + f'{resolution_V_display:.4f} '
        + q_unit
        + ', '

        r'$\delta Q_W$ = '
        + f'{resolution_W_display:.4f} '
        + q_unit
        + ', '

        f'δℏω = {resolution_energy:.4f}'
        + r' (meV)',

        fontsize=11,
        y=0.98
    )

    # ============================================================
    # U vs E
    # ============================================================

    ax1.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )

    ax1.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )

    ax1.set_xlabel(
        rf"$\delta Q_U$ {q_unit}"
    )

    ax1.set_ylabel(
        "δℏω (meV)"
    )

    ax1.set_title(
        r"$\delta Q_U$ vs $\hbar\omega$ ellipse",
        fontsize=12
    )

    ax1.set_xlim(
        [-Urange_lim / normQx, Urange_lim / normQx]
    )

    ax1.set_ylim(
        [-Zrange_lim, Zrange_lim]
    )

    ax1.grid(True)


    # ============================================================
    # V vs E
    # ============================================================

    ax2.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )

    ax2.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )

    ax2.set_xlabel(
        rf"$\delta Q_V$ {q_unit}"
    )

    ax2.set_ylabel(
        "δℏω (meV)"
    )

    ax2.set_title(
        r"$\delta Q_V$ vs $\hbar\omega$ ellipse",
        fontsize=12
    )

    ax2.set_xlim(
        [-Vrange_lim / normQy, Vrange_lim / normQy]
    )

    ax2.set_ylim(
        [-Zrange_lim, Zrange_lim]
    )

    ax2.grid(True)

    # ============================================================
    # U-V projection
    # ============================================================

    A_UV, B_UV, C_UV, D_UV, E_UV, F_UV = \
        ellipse_coefficients(
            RM_U,
            log2=np.log(2),
            plane=("x", "y")
        )


    # ============================================================
    # U-V slice
    # ============================================================

    A_UV_s, B_UV_s, C_UV_s, D_UV_s, E_UV_s, F_UV_s = \
        ellipse_slice_coefficients(
            RM_U,
            ("x", "y")
        )


    # ============================================================
    # U-V 楕円を描画する関数
    #
    # 今回はまず直交座標として表示する。
    # ============================================================

    def plot_ellipse_uv(
        A,
        B,
        C,
        D,
        E,
        F,
        Urange_lim,
        Vrange_lim,
        ax,
        labels,
        color,
        ls,
        shift_x=0,
        shift_y=0
    ):

        U = np.linspace(
            -Urange_lim,
            Urange_lim,
            100
        )

        V = np.linspace(
            -Vrange_lim,
            Vrange_lim,
            100
        )

        U_grid, V_grid = np.meshgrid(U, V)

        ellipse = (
            A * U_grid**2
            + B * U_grid * V_grid
            + C * V_grid**2
            + D * U_grid
            + E * V_grid
            + F
        )

        U_shifted = U_grid + shift_x
        V_shifted = V_grid + shift_y

        ax.contour(
            U_shifted,
            V_shifted,
            ellipse,
            levels=[0],
            colors=color,
            label=labels,
            linestyles=ls
        )


    # ============================================================
    # U-V projection ellipse
    # 実線
    # ============================================================

    plot_ellipse_uv(
        A_UV,
        B_UV,
        C_UV,
        D_UV,
        E_UV,
        F_UV,
        Urange_lim,
        Vrange_lim / np.abs(np.sin(uv_angle)),
        ax3,
        labels="",
        color="black",
        ls=["-"],
        shift_x=0,
        shift_y=0
    )


    # ============================================================
    # U-V slice ellipse
    # 破線
    # ============================================================

    plot_ellipse_uv(
        A_UV_s,
        B_UV_s,
        C_UV_s,
        D_UV_s,
        E_UV_s,
        F_UV_s,
        Urange_lim,
        Vrange_lim / np.abs(np.sin(uv_angle)),
        ax3,
        labels="",
        color="black",
        ls=["--"],
        shift_x=0,
        shift_y=0
    )


    # ============================================================
    # U-V 座標軸
    # ============================================================

    ax3.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )

    ax3.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.5
    )


    # ============================================================
    # U-V 軸ラベル
    # ============================================================

    ax3.set_xlabel(
        r"$\delta Q_U$ $(\mathrm{\AA}^{-1})$"
    )

    ax3.set_ylabel(
        r"$\delta Q_V$ $(\mathrm{\AA}^{-1})$"
    )

    ax3.set_title(
        r"$\delta Q_U$ vs $\delta Q_V$ ellipse",
        fontsize=12
    )

    # ============================================================
    # 実際の V 方向を示す補助線
    # ============================================================

    V_line = Vrange_lim

    ax3.plot(
        [
            -V_line * np.cos(uv_angle),
                V_line * np.cos(uv_angle)
        ],
        [
            -V_line * np.sin(uv_angle),
                V_line * np.sin(uv_angle)
        ],
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=3
    )

    # ============================================================
    # 実際の U 方向を示す補助線
    # ============================================================

    U_line = Urange_lim

    ax3.plot(
        [
            -U_line,
            U_line
        ],
        [
            0,
            0
        ],
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=3
    )

    # ============================================================
    # 表示範囲
    # ============================================================

    ax3.set_xlim(
        [-Urange_lim, Urange_lim]
    )

    ax3.set_ylim(
        [-Vrange_lim / np.abs(np.sin(uv_angle)), Vrange_lim / np.abs(np.sin(uv_angle))]
    )


    # ============================================================
    # 通常の直交グリッド
    # ============================================================

    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')

    # ============================================================
    # W vs E
    # ============================================================

    ax4.axhline(
        0,
        color="green",
        linestyle="--",
        linewidth=0.5
    )

    ax4.axvline(
        0,
        color="green",
        linestyle="--",
        linewidth=0.5
    )

    ax4.set_xlabel(
        rf"$\delta Q_W$ {q_unit}"
    )

    ax4.set_ylabel(
        "δℏω (meV)"
    )

    ax4.set_title(
        r"$\delta Q_W$ vs $\hbar\omega$ ellipse",
        fontsize=12
    )

    ax4.set_xlim(
        [-Wrange_lim / normQz, Wrange_lim / normQz]
    )

    ax4.set_ylim(
        [-Zrange_lim, Zrange_lim]
    )

    ax4.grid(True)


    # ============================================================
    # return
    # ============================================================

    return RM, fig