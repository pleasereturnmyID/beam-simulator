"""sweep.py - 시뮬레이션 데이터셋 생성과 분석 (Google Colab에서도 그대로 실행 가능)

실행: python sweep.py   → sweep.csv 저장 + 분석 결과 출력
Colab: beam.py와 sweep.py를 업로드한 뒤 `!python sweep.py`
"""
import itertools

import numpy as np
import pandas as pd

from beam import find_mmax, reactions, self_weight

MATERIALS = {"none": 0, "wood": 500, "concrete": 2400, "steel": 7850}  # kg/m^3
B, H = 0.20, 0.40  # 단면 고정 [m]


def make_sweep():
    """L 2~8 m(7) × a/L 0.1~0.9(9) × P 5~30 kN(6) × 재료 4종 = 1,512행"""
    rows = []
    for L, ar, P, (mat, rho) in itertools.product(
            range(2, 9), np.round(np.arange(0.1, 1.0, 0.1), 1),
            range(5000, 30001, 5000), MATERIALS.items()):
        a = L * ar
        w = self_weight(rho, B, H)
        RA, RB = reactions(L, a, P, w)
        xm, M = find_mmax(L, a, P, w)
        rows.append({"L_m": L, "a_ratio": ar, "a_m": a, "P_N": P, "material": mat, "rho": rho,
                     "w_N_m": w, "R_A_N": RA, "R_B_N": RB, "x_Mmax_m": xm, "M_max_Nm": M,
                     "selfweight_share": (w * L * L / 8) / M})
    return pd.DataFrame(rows)


def analyze(df):
    print("행 수:", len(df), "| 결측:", int(df.isna().sum().sum()), "| 중복:", int(df.duplicated().sum()))
    print("\n[기술통계] Mmax (N·m)\n", df["M_max_Nm"].describe().round(1))
    print("\n[a/L별 평균 Mmax]\n", df.groupby("a_ratio")["M_max_Nm"].mean().round(0))
    print("\n[Pearson 상관, Mmax와]\n", df[["L_m", "a_ratio", "P_N", "rho"]].corrwith(df["M_max_Nm"]).round(3))
    print("\n[재료별 평균 자중 비중]\n", df.groupby("material")["selfweight_share"].mean().round(2))
    shift = (df["x_Mmax_m"] - df["a_m"]).abs() > 1e-9
    print("\n[Mmax 위치가 하중점에서 벗어난 비율]\n", shift.groupby(df["material"]).mean().round(3))

    z = df[df["rho"] == 0]
    X = np.column_stack([np.ones(len(z)), np.log(z.P_N), np.log(z.a_m), np.log(z.L_m - z.a_m), np.log(z.L_m)])
    beta = np.linalg.lstsq(X, np.log(z.M_max_Nm), rcond=None)[0]
    print("\n[로그 선형회귀, ρ=0] (β0, logP, log a, log(L-a), log L) =", beta.round(3))

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        X = df[["L_m", "a_ratio", "P_N", "rho"]]
        Xtr, Xte, ytr, yte = train_test_split(X, df["M_max_Nm"], test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
        print("\n[랜덤포레스트] 테스트 R² =", round(r2_score(yte, rf.predict(Xte)), 3))
        print("변수 중요도:", dict(zip(X.columns, rf.feature_importances_.round(3))))
    except ImportError:
        print("\n(scikit-learn이 없어 랜덤포레스트는 건너뜀)")


if __name__ == "__main__":
    data = make_sweep()
    data.to_csv("sweep.csv", index=False, encoding="utf-8-sig")
    print("sweep.csv 저장 완료")
    analyze(data)
