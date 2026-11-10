"""beam.py - 단순보(왼쪽 핀, 오른쪽 롤러) + 집중하중 1개 + 자중(등분포) 해석 엔진.

단위: 길이 m, 힘 N, 모멘트 N·m, 밀도 kg/m^3
부호: 위로 향한 반력 +, 아래로 처지는(sagging) 모멘트 +
streamlit을 import하지 않으므로 Colab, pytest에서 그대로 쓸 수 있다.
"""
import numpy as np

G = 9.81  # m/s^2


def self_weight(rho, b, h):
    """자중 등분포하중 w [N/m] = ρ·b·h·g"""
    return rho * b * h * G


def reactions(L, a, P, w):
    """ΣM_A = 0, ΣFy = 0 으로 구한 반력 (R_A, R_B) [N]"""
    R_B = (P * a + w * L ** 2 / 2) / L
    R_A = P + w * L - R_B
    return R_A, R_B


def shear(x, L, a, P, w):
    """전단력 V(x) [N]. x = a 에서는 왼쪽 값 V(a-)를 쓴다."""
    R_A, _ = reactions(L, a, P, w)
    x = np.asarray(x, dtype=float)
    return R_A - w * x - P * (x > a)


def moment(x, L, a, P, w):
    """굽힘 모멘트 M(x) [N·m]"""
    R_A, _ = reactions(L, a, P, w)
    x = np.asarray(x, dtype=float)
    return R_A * x - w * x ** 2 / 2 - P * np.clip(x - a, 0.0, None)


def find_mmax(L, a, P, w):
    """dM/dx = V 이므로 V가 0이 되거나 부호가 바뀌는 점에서 Mmax. 위치를 해석적으로 찾는다."""
    R_A, _ = reactions(L, a, P, w)
    if w > 0 and R_A / w < a:              # 왼쪽 구간(0~a)에서 V = 0
        x_m = R_A / w
    elif w == 0 or R_A - P - w * a <= 0:    # 하중점에서 V의 부호가 바뀜
        x_m = a
    else:                                   # 오른쪽 구간(a~L)에서 V = 0
        x_m = (R_A - P) / w
    x_m = min(max(x_m, 0.0), L)
    return x_m, float(moment(x_m, L, a, P, w))


def solve_beam(L, rho, b, h, a, P, n=1001):
    """전체 해석. 입력이 범위를 벗어나면 ValueError."""
    if not (L > 0 and 0 <= a <= L and P >= 0 and rho >= 0 and b > 0 and h > 0):
        raise ValueError("입력 범위 오류: L>0, 0≤a≤L, P≥0, ρ≥0, b·h>0 이어야 합니다.")
    w = self_weight(rho, b, h)
    R_A, R_B = reactions(L, a, P, w)
    # x = a 를 양쪽 구간에 모두 넣어 SFD의 수직 점프를 정확히 그린다
    k = max(2, int(round(n * a / L))) if L > 0 else 2
    x_left = np.linspace(0, a, k)
    x_right = np.linspace(a, L, max(2, n - k))
    x = np.concatenate([x_left, x_right])
    V = np.concatenate([R_A - w * x_left, R_A - w * x_right - P])
    M = moment(x, L, a, P, w)
    x_m, M_max = find_mmax(L, a, P, w)
    return {"w_N_m": w, "R_A_N": R_A, "R_B_N": R_B,
            "x_Mmax_m": x_m, "M_max_Nm": M_max,
            "x": x, "V": V, "M": M}


def result_summary(r, L, rho, b, h, a, P):
    """화면 표시와 Claude 해설에 쓰는 출력 JSON (배열 제외, 입력값 포함)"""
    return {"L_m": round(L, 3), "P_N": round(P, 2), "rho_kg_m3": round(rho, 1),
            "b_m": round(b, 3), "h_m": round(h, 3),
            "w_N_m": round(r["w_N_m"], 2), "R_A_N": round(r["R_A_N"], 2),
            "R_B_N": round(r["R_B_N"], 2), "x_Mmax_m": round(r["x_Mmax_m"], 3),
            "M_max_Nm": round(r["M_max_Nm"], 2), "a_m": round(a, 3),
            "mmax_at_load_point": abs(r["x_Mmax_m"] - a) < 1e-9}
