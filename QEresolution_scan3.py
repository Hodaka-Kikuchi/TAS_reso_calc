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

from PIL import Image  # GIF 保存のために必要

def calcresolution_scan3(apr_value,sense,astar,bstar,cstar,sv1,sv2,sv3,A_sets,QE_sets,Ni_mir,bpe,fixe,focus_cond,inst_param,entry_values,initial_index=0,save_gif=False,gif_name="resolution.gif"):
    # save_gifがTrueだと保存、Falseだと非保存
    '''
    # INIファイルから設定を読み込む
    config = configparser.ConfigParser()
    # .exe化した場合に対応する
    if getattr(sys, 'frozen', False):
        # .exeの場合、sys.argv[0]が実行ファイルのパスになる
        ini_path = os.path.join(os.path.dirname(sys.argv[0]), 'config.ini')
    else:
        # .pyの場合、__file__がスクリプトのパスになる
        ini_path = os.path.join(os.path.dirname(__file__), 'config.ini')

    config.read(ini_path)
    '''
    # divergenceの読み出し
    div_1st_m = float(entry_values.get("div_1st_m"))
    div_1st_h = float(entry_values.get("div_1st_h"))
    div_1st_v = float(entry_values.get("div_1st_v"))
    div_2nd_h = float(entry_values.get("div_2nd_h"))
    div_2nd_v = float(entry_values.get("div_2nd_v"))
    div_3rd_h = float(entry_values.get("div_3rd_h"))
    div_3rd_v = float(entry_values.get("div_3rd_v"))
    div_4th_h = float(entry_values.get("div_4th_h"))
    div_4th_v = float(entry_values.get("div_4th_v"))
    
    mos_mono_h = float(entry_values.get("mos_mono_h"))
    mos_mono_v = float(entry_values.get("mos_mono_v"))
    mos_sam_h = float(entry_values.get("mos_sam_h"))
    mos_sam_v = float(entry_values.get("mos_sam_v"))
    mos_ana_h = float(entry_values.get("mos_ana_h"))
    mos_ana_v = float(entry_values.get("mos_ana_v"))

    # focusing conditionの読み出し
    MHF = focus_cond["MHF"]
    MVF = focus_cond["MVF"]
    AHF = focus_cond["AHF"]
    AVF = focus_cond["AVF"]

    num_mono_h = focus_cond["num_mono_h"]
    num_mono_v = focus_cond["num_mono_v"]
    num_ana_h  = focus_cond["num_ana_h"]
    num_ana_v  = focus_cond["num_ana_v"]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2グリッドのサブプロット作成
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.3, hspace=0.4)

    # サブプロットの指定：左上 (0, 0)、右上 (0, 1)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    # スライダー設定（描画領域調整が必要）
    ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'scan number', 1, len(A_sets), valinit=initial_index + 1, valstep=1)
    
    # フレーム保存用リスト
    frames = []

    # 楕円描画関数の中で ax1, ax2 に描くように変更
    """
    # 例: Q_parallel を ax1 に、Q_perp を ax2 に描画する
    ax1.set_title("$Q_{\\parallel}$ vs ℏω ellipse", fontsize=12)
    ax2.set_title("$Q_{\\perp}$ vs ℏω ellipse", fontsize=12)
    ax3.set_title("$Q_{\\parallel}$ vs $Q_{\\perp}$ ellipse", fontsize=12)
    ax4.set_title("$Q_{out of plane}$ vs ℏω ellipse", fontsize=12)
    """
    # 初期タイトル
    plt.suptitle(
        f'ℏω: {QE_sets[initial_index][0]} meV, h: {QE_sets[initial_index][1]}, k: {QE_sets[initial_index][2]}, l: {QE_sets[initial_index][3]}',
        fontsize=12
    )
    
    # ここでscanの最初と最後のポイントの分解能の計算する。
    # 空リストに格納
    X_vals, Y_vals, Z_vals, W_vals = [], [], [], []
    for index in [0, -1]:
        A1, A2, A3 = A_sets[index]
        hw = QE_sets[index][0]
        
        if fixe==0: # ei fix
            Ei = bpe
            Ef = bpe - hw
        elif fixe==1: # ef fix
            Ei = bpe + hw
            Ef = bpe

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
        if Ni_mir == 0:
            alpha1 = div_1st_h / 60 / 180 * pi * 0.4246609
            beta1 = div_1st_v / 60 / 180 * pi * 0.4246609 
        elif Ni_mir == 1:
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
        if AHF==0:
            alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609
        elif AHF==1:
            L=inst_param["sample_to_ana"]
            W=inst_param["ana_width"]*num_ana_h*np.sin(np.radians(A3))
            af=2 * np.degrees(np.arctan((W / 2) / L))
            #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
            alpha3 = af / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
            
            #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
        
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
        if apr_value == "P":
            beam_width = inst_param["beam_width"]
            beam_height = inst_param["beam_height"]
            # ==== Beam shape ====
            beamw = (beam_width)**2
            beamh = (beam_height)**2
            bshape = np.diag([beamw, beamh])

            # ==== Mono shape ====
            mono_width = inst_param["mono_width"]
            mono_height = inst_param["mono_height"]
            mono_depth = inst_param["mono_depth"]
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
            ana_width = inst_param["ana_width"]
            ana_height = inst_param["ana_height"]
            ana_depth = inst_param["ana_depth"]
            anaw = (num_ana_h*ana_width)**2
            anah = (num_ana_v*ana_height)**2
            anad = (ana_depth)**2
            ashape = np.diag([anad, anaw, anah])

            # ==== det shape ====
            det_width = inst_param["det_width"]
            det_height = inst_param["det_height"]
            detw = (det_width)**2
            deth = (det_height)**2
            dshape = np.diag([detw, deth])

            # ==== S matrix ====
            Sinv = block_diag(bshape, mshape, sshape, ashape, dshape)  # S^-1
            S = np.linalg.inv(Sinv)

            # ==== Distances ====
            L0 = inst_param["source_to_mono"]
            L1 = inst_param["mono_to_sample"]
            #L1mon = 1 # only flux normalization
            L2 = inst_param["sample_to_ana"]
            L3 = inst_param["ana_to_det"]

            def focusing_curvature(L_1, L_2, theta):
                # 有効焦点距離。ただし、単位をmmからmに直す必要がある。
                f = 1.0 / (1.0/L_1 + 1.0/L_2)

                # θ をラジアンに変換
                theta_rad = np.radians(theta)

                # 曲率 R = 2*f*|sin(theta)|
                R = 2.0 * f * np.abs(np.sin(theta_rad))

                return R

            # focusing:
            if MVF == 0:
                monorv = 1e10
            elif MVF ==1:
                monorv = focusing_curvature(L0,L1,thetaM)
            if MHF == 0:
                monorh = 1e10
            elif MHF ==1:
                monorh = focusing_curvature(L0,L1,thetaM)
            if AVF == 0:
                anarv = 1e10
            elif AVF ==1:
                anarv = focusing_curvature(L2,L3,thetaA)
            if AHF == 0:
                anarh = 1e10
            elif AHF ==1:
                anarh = focusing_curvature(L2,L3,thetaA)

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
        if apr_value == "P":
            Minv = (B @ A @ np.linalg.inv(np.linalg.inv(D @ np.linalg.inv(S + T.T @ F @ T) @ D.T) + G) @ A.T @ B.T)
        elif apr_value == "CN":
            if AHF == 1:
                P = np.linalg.inv(HF)
                P[4, 4] = (1 / (kf * alpha3)) ** 2
                P[3, 4] = 0
                P[3, 3] = (np.tan(np.radians(thetaA)) / (etaA * kf)) ** 2
                P[4, 3] = 0
                Pinv = np.linalg.inv(P)
                Minv = B @ Pinv @ B.T
            elif AHF == 0:
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
        
        # 座標変換
        """
        # 古いコード
        Qx = sv1[0]*astar+sv1[1]*bstar+sv1[2]*cstar
        Qy = sv2[0]*astar+sv2[1]*bstar+sv2[2]*cstar
        Qvect = QE_sets[index][1]*astar+QE_sets[index][2]*bstar+QE_sets[index][3]*cstar
        
        # 行列を作成して連立方程式を解く
        M = np.column_stack((Qx, Qy))  # 3x2 行列
        A_B, residuals, rank, s = np.linalg.lstsq(M, Qvect, rcond=None)
        A, B = A_B

        # 角度の計算(ラジアン)
        theta_rad = np.arctan2(B, A)
        # 散乱面内で回転。
        rot_mat = np.array([
                            [np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
                            [np.sin(theta_rad),  np.cos(theta_rad), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
        """

        Qx = sv1[0]*astar+sv1[1]*bstar+sv1[2]*cstar
        Qy = sv2[0]*astar+sv2[1]*bstar+sv2[2]*cstar
        Qz = sv3[0]*astar+sv3[1]*bstar+sv3[2]*cstar
        Qvect = QE_sets[index][1]*astar+QE_sets[index][2]*bstar+QE_sets[index][3]*cstar
        
        # Q方向の単位ベクトル
        uq = Qvect / np.linalg.norm(Qvect)

        # MATLABの scalar(u,v) に対応 → 普通の内積でOK (すでに直交座標系に変換済みなので)
        xq = np.dot(Qx, uq) / np.linalg.norm(Qx)
        yq = np.dot(Qy, uq) / np.linalg.norm(Qy)

        # 回転角度
        theta_rad = np.arctan2(yq, xq)

        # 回転行列（MATLABの tmat と同じ形）
        rot_mat = np.array([
            [ np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
            [ np.sin(theta_rad),  np.cos(theta_rad), 0, 0],
            [0,                  0,                  1, 0],
            [0,                  0,                  0, 1]
        ])
        
        # 相似変換
        RM = rot_mat @ RM @ rot_mat.T
        
        if sense == '-+-':
            # 上下反転
            # rightではそのまま、leftでaxis2をaxis1に対してミラーさせる。
            S = np.diag([1.0, -1.0, 1.0, 1.0])   # y軸のみ反転
            RM_flipped = S @ RM @ S.T
            
            RM = RM_flipped
        elif sense == '+-+':
            pass

        # RMは(q//,q⊥,hw,qz)における空間分布
        # これを(qx(axis1),qy(axis2),hw,qz)に置ける空間分布に変換する。
        
        # プロット範囲
        #Xrange_lim = 0.1
        #Zrange_lim = 0.5
        
        # Qx=Q//,Qy=Q⊥の定義
        
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
        max_w, coords_w = find_max_along_axis(RM, axis="w")# E
            
        X_vals.append(max_x)
        Y_vals.append(max_y)
        Z_vals.append(max_z)
        W_vals.append(max_w)

    # 大きい方を採用
    Xrange_lim = max(X_vals) * 1.25
    Yrange_lim = max(Y_vals) * 1.25
    Zrange_lim = max(Z_vals) * 1.25
    Wrange_lim = max(W_vals) * 1.25
    
    def update(val):
        nonlocal div_1st_h, div_1st_v  # ← これを追加
        index = int(val)-1
        A1, A2, A3 = A_sets[index]
        hw = QE_sets[index][0]
        
        # プロットを再描画
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        if fixe==0: # ei fix
            Ei = bpe
            Ef = bpe - hw
        elif fixe==1: # ef fix
            Ei = bpe + hw
            Ef = bpe

        ki=(Ei/2.072)**(1/2)
        kf=(Ef/2.072)**(1/2)
        Q = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(np.radians(A2))) 
        
        # C1とA1の計算
        C1 = A1/2

        # C3とA3の計算
        C3 = A3/2

        # A2は既に負の値, reslibとは異なる定義
        thetaM = C1
        thetaA = C3
        thetaS = A2 / 2
        
        phi = np.degrees(np.arctan2((-kf * np.sin(np.radians(2 * thetaS))), (ki - kf * np.cos(np.radians(2 * thetaS)))))

        # Define constants for the resolution matrices
        # ここでαi とβi は、コリメータの水平方向と鉛直方向における発散角を表している。η とη′をモノクロメータとアナライザの水平方向と鉛直方向のモザイクのFWHM

        #theta0 = 0.1 #(A^-1)
        #lamda = (81.81 / Ei)**(1/2)
        # 0.4246609 = 1/(2*sqrt(2*log(2)))
        if Ni_mir == 0:
            alpha1 = div_1st_h / 60 / 180 * pi * 0.4246609
            beta1 = div_1st_v / 60 / 180 * pi * 0.4246609 
        elif Ni_mir == 1:
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
        if AHF==0:
            alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609
        elif AHF==1:
            L=inst_param["sample_to_ana"]
            W=inst_param["ana_width"]*num_ana_h*np.sin(np.radians(A3))
            af=2 * np.degrees(np.arctan((W / 2) / L))
            #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
            alpha3 = af / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
            
            #alpha3 = div_3rd_h / 60 / 180 * pi * 0.4246609 * (8*np.log(2)/12)**(1/2)
        
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
        if apr_value == "P":
            beam_width = inst_param["beam_width"]
            beam_height = inst_param["beam_height"]
            # ==== Beam shape ====
            beamw = (beam_width)**2
            beamh = (beam_height)**2
            bshape = np.diag([beamw, beamh])

            # ==== Mono shape ====
            mono_width = inst_param["mono_width"]
            mono_height = inst_param["mono_height"]
            mono_depth = inst_param["mono_depth"]
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
            ana_width = inst_param["ana_width"]
            ana_height = inst_param["ana_height"]
            ana_depth = inst_param["ana_depth"]
            anaw = (num_ana_h*ana_width)**2
            anah = (num_ana_v*ana_height)**2
            anad = (ana_depth)**2
            ashape = np.diag([anad, anaw, anah])

            # ==== det shape ====
            det_width = inst_param["det_width"]
            det_height = inst_param["det_height"]
            detw = (det_width)**2
            deth = (det_height)**2
            dshape = np.diag([detw, deth])

            # ==== S matrix ====
            Sinv = block_diag(bshape, mshape, sshape, ashape, dshape)  # S^-1
            S = np.linalg.inv(Sinv)

            # ==== Distances ====
            L0 = inst_param["source_to_mono"]
            L1 = inst_param["mono_to_sample"]
            #L1mon = 1 # only flux normalization
            L2 = inst_param["sample_to_ana"]
            L3 = inst_param["ana_to_det"]

            def focusing_curvature(L_1, L_2, theta):
                # 有効焦点距離。ただし、単位をmmからmに直す必要がある。
                f = 1.0 / (1.0/L_1 + 1.0/L_2)

                # θ をラジアンに変換
                theta_rad = np.radians(theta)

                # 曲率 R = 2*f*|sin(theta)|
                R = 2.0 * f * np.abs(np.sin(theta_rad))

                return R

            # focusing:
            if MVF == 0:
                monorv = 1e10
            elif MVF ==1:
                monorv = focusing_curvature(L0,L1,thetaM)
            if MHF == 0:
                monorh = 1e10
            elif MHF ==1:
                monorh = focusing_curvature(L0,L1,thetaM)
            if AVF == 0:
                anarv = 1e10
            elif AVF ==1:
                anarv = focusing_curvature(L2,L3,thetaA)
            if AHF == 0:
                anarh = 1e10
            elif AHF ==1:
                anarh = focusing_curvature(L2,L3,thetaA)

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
        if apr_value == "P":
            Minv = (B @ A @ np.linalg.inv(np.linalg.inv(D @ np.linalg.inv(S + T.T @ F @ T) @ D.T) + G) @ A.T @ B.T)
        elif apr_value == "CN":
            if AHF == 1:
                P = np.linalg.inv(HF)
                P[4, 4] = (1 / (kf * alpha3)) ** 2
                P[3, 4] = 0
                P[3, 3] = (np.tan(np.radians(thetaA)) / (etaA * kf)) ** 2
                P[4, 3] = 0
                Pinv = np.linalg.inv(P)
                Minv = B @ Pinv @ B.T
            elif AHF == 0:
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
        
        # 座標変換
        """
        Qx = sv1[0]*astar+sv1[1]*bstar+sv1[2]*cstar
        Qy = sv2[0]*astar+sv2[1]*bstar+sv2[2]*cstar
        Qvect = QE_sets[index][1]*astar+QE_sets[index][2]*bstar+QE_sets[index][3]*cstar
        
        # 行列を作成して連立方程式を解く
        M = np.column_stack((Qx, Qy))  # 3x2 行列
        A_B, residuals, rank, s = np.linalg.lstsq(M, Qvect, rcond=None)
        A, B = A_B

        # 角度の計算(ラジアン)
        theta_rad = np.arctan2(B, A)
        print(theta_rad/pi*180)
        
        # 既に象限判定される
        if A>=0 and B>=0:
            theta_rad = theta_rad
        elif A<=0 and B>=0:
            theta_rad = theta_rad+90
        elif A<=0 and B<=0:
            theta_rad = -(theta_rad+90)
        elif A>=0 and B<=0:
            theta_rad = -theta_rad
        
        # 散乱面内で回転。
        rot_mat = np.array([
                            [np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
                            [np.sin(theta_rad),  np.cos(theta_rad), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
                        
        # 相似変換
        RM = rot_mat @ RM @ rot_mat.T
        """
        Qvect = QE_sets[index][1]*astar+QE_sets[index][2]*bstar+QE_sets[index][3]*cstar
    
        # Q方向の単位ベクトル
        uq = Qvect / np.linalg.norm(Qvect)

        # MATLABの scalar(u,v) に対応 → 普通の内積でOK (すでに直交座標系に変換済みなので)
        xq = np.dot(Qx, uq) / np.linalg.norm(Qx)
        yq = np.dot(Qy, uq) / np.linalg.norm(Qy)

        # 回転角度
        theta_rad = np.arctan2(yq, xq)
        
        #print(theta_rad/pi*180)
        # 回転行列（MATLABの tmat と同じ形）
        rot_mat = np.array([
            [ np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
            [ np.sin(theta_rad),  np.cos(theta_rad), 0, 0],
            [0,                  0,                  1, 0],
            [0,                  0,                  0, 1]
        ])
        
        # 相似変換
        RM = rot_mat @ RM @ rot_mat.T
        if sense == '-+-':
            # 上下反転
            # rightではそのまま、leftでaxis2をaxis1に対してミラーさせる。
            S = np.diag([1.0, -1.0, 1.0, 1.0])   # y軸のみ反転
            RM_flipped = S @ RM @ S.T
            
            RM = RM_flipped
        elif sense == '+-+':
            pass
        # RMは(q//,q⊥,hw,qz)における空間分布
        # これを(qx(axis1),qy(axis2),hw,qz)に置ける空間分布に変換する。
        """
        # 保存するリスト
        data_list = []

        # ヘッダーと行列を順に追加する関数
        def add_matrix_to_list(name, matrix):
            data_list.append([f"=== {name} ==="])
            for row in matrix:
                data_list.append(list(row))
            data_list.append([])  # 空行を挿入

        # 追加していく
        add_matrix_to_list('A', A)
        add_matrix_to_list('B', B)
        add_matrix_to_list('C', C)
        #add_matrix_to_list('D', D)
        add_matrix_to_list('G', G)
        add_matrix_to_list('F', F)
        add_matrix_to_list('HF', HF)
        add_matrix_to_list('M', M)
        add_matrix_to_list('Minv', Minv)
        add_matrix_to_list('RM', RM)

        # DataFrameに変換（短い行にはNaNが入る）
        df = pd.DataFrame(data_list)

        # CSV出力
        df.to_csv('matrices_output.csv', index=False, header=False)
        """
        # プロット範囲
        #Xrange_lim = 0.1
        #Zrange_lim = 0.5
        
        # Qx=Q//,Qy=Q⊥の定義
        
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
            x = np.linspace(-Xrange_lim, Xrange_lim, 500)
            z = np.linspace(-Zrange_lim, Zrange_lim, 500)
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
            x = np.linspace(-Xrange_lim, Xrange_lim, 500)
            z = np.linspace(-Zrange_lim, Zrange_lim, 500)
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
            x = np.linspace(-Xrange_lim, Xrange_lim, 500)
            z = np.linspace(-Zrange_lim, Zrange_lim, 500)
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
            x = np.linspace(-Wrange_lim, Wrange_lim, 500)
            z = np.linspace(-Zrange_lim, Zrange_lim, 500)
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
            f'ℏω: {QE_sets[index][0]} meV, h: {QE_sets[index][1]}, k: {QE_sets[index][2]}, l: {QE_sets[index][3]}\n'
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
        
        # フレーム保存（GIF用）
        if save_gif:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(image))

        # プロットの表示
        plt.draw()
        
    slider.on_changed(update)

    # 初期の描画を行う
    update(slider.val)
    
    def on_key(event):
        """
        左右矢印キーでスライダーを動かす
        """
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, len(A_sets)))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, 1))

    # キーイベントを設定
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    if save_gif:
        for val in range(1, len(A_sets) + 1):
            slider.set_val(val)
        frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"GIF 保存完了: {gif_name}")