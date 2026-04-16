import math
import numpy as np

def UB_calc(sv1, sv2, astar, bstar, cstar, alpha_star, beta_star, gamma_star, n_a, n_b, n_c, a, b, c, alpha, beta, gamma):
    """UBマトリックスとその構成要素を計算し、辞書で返す"""

    # ベクトル u1, u2 の計算
    u1 = sv1[0] * astar + sv1[1] * bstar + sv1[2] * cstar
    U1 = u1 / np.linalg.norm(u1)
    
    u2 = sv2[0] * astar + sv2[1] * bstar + sv2[2] * cstar
    uu2 = u2 - np.dot(U1, u2) * U1
    U2 = uu2 / np.linalg.norm(uu2)
    
    # U3の計算 (U1とU2の外積)
    U3 = np.cross(U1, U2)
    #Un = U行列のn行目(横)
    
    # Uマトリックス
    U = np.vstack([U1, U2, U3])
    
    # Bマトリックス
    B = 1 / (2 * math.pi) * np.array([
        [n_a, n_b * np.cos(np.radians(gamma_star)), n_c * np.cos(np.radians(beta_star))],
        [0, n_b * np.sin(np.radians(gamma_star)), -n_c * np.sin(np.radians(beta_star)) * np.cos(np.radians(alpha))],
        [0, 0, 2 * math.pi / c]
    ])
    B[np.abs(B) <= 1e-6] = 0 #超重要,他のものにも適応
    
    # UBマトリックス
    UB = U @ B
    UB[np.abs(UB) <= 1e-6] = 0 #超重要,他のものにも適応
    
    # 結果を辞書としてまとめる
    result = {
        'sv1' : sv1,
        'sv2' : sv2,
        'U': U,
        'B': B,
        'UB': UB
    }
    
    return result
