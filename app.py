"""app.py - 집중하중이 가해지는 단순보의 내부력 시뮬레이터
실행: streamlit run app.py
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from beam import result_summary, solve_beam
from sweep import make_sweep

st.set_page_config(page_title="보 내부력 시뮬레이터", layout="wide")
st.title("집중하중이 가해지는 단순보의 내부력 시뮬레이터")
st.caption("왼쪽 핀(힌지) · 오른쪽 롤러 단순보 | 단위: m, N, N·m | 계산은 파이썬, 문장 해석과 해설은 Claude")

DEFAULTS = {"L": 6.0, "rho": 7850, "b_cm": 20, "h_cm": 40, "a": 2.0, "a_keep": 2.0, "P": 10000}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_text():
    """[F7] 문장 → Claude → 검증 → 슬라이더 값 덮어쓰기 (콜백 안이라 위젯 값 변경 가능)"""
    from ai import parse_problem
    params, msgs = parse_problem(st.session_state.get("problem_text", ""))
    if params:
        new = dict(L=round(params["L_m"] * 20) / 20, a=round(params["a_m"] * 20) / 20,
                   P=int(round(params["P_N"], -1)), rho=int(round(params["rho_kg_m3"], -1)),
                   b_cm=int(round(params["b_m"] * 100)), h_cm=int(round(params["h_m"] * 100)))
        new["a"] = min(new["a"], new["L"])
        new["a_keep"] = new["a"]
        pairs = [("L", new["L"], params["L_m"], "m"), ("a", new["a"], params["a_m"], "m"),
                 ("P", new["P"], params["P_N"], "N"), ("ρ", new["rho"], params["rho_kg_m3"], "kg/m³"),
                 ("b", new["b_cm"] / 100, params["b_m"], "m"), ("h", new["h_cm"] / 100, params["h_m"], "m")]
        for name, snapped, orig, unit in pairs:
            if abs(snapped - orig) > 1e-9:
                msgs.append(f"{name}: {orig:g} → {snapped:g} {unit} (슬라이더 눈금에 맞춰 반올림)")
        st.session_state.update(new)
        msgs.insert(0, "문장의 조건을 슬라이더에 반영했습니다.")
    st.session_state["parse_msgs"] = msgs
    st.session_state.pop("explain", None)


def draw(L, a, P, r):
    x, V, M = r["x"], r["V"], r["M"]
    w, RA, RB, xm, Mmax = r["w_N_m"], r["R_A_N"], r["R_B_N"], r["x_Mmax_m"], r["M_max_Nm"]
    fig, (ax0, axV, axM) = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True,
                                        gridspec_kw={"height_ratios": [1.1, 1.4, 1.4]})
    # 보 모형: 사각형 보 + 핀(삼각형) + 롤러(원) + 하중 화살표 + 자중 화살표
    s = 0.025 * L
    ax0.add_patch(patches.Rectangle((0, 0), L, 0.25, fc="#d9d9d9", ec="black"))
    ax0.add_patch(patches.Polygon([[0, 0], [-s, -0.3], [s, -0.3]], fc="white", ec="black"))
    ax0.add_patch(patches.Ellipse((L, -0.15), 2 * s, 0.3, fc="white", ec="black"))
    if w > 0:
        for xi in np.linspace(0, L, 13):
            ax0.annotate("", xy=(xi, 0.25), xytext=(xi, 0.5),
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.8))
        ax0.text(L / 2, 0.55, f"self-weight w = {w:,.0f} N/m", ha="center", fontsize=8, color="gray",
                 bbox=dict(fc="white", ec="none", pad=1))
    ax0.annotate("", xy=(a, 0.25), xytext=(a, 1.15), arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax0.text(a, 1.2, f"P = {P:,} N", ha="center", color="red")
    ax0.text(0, -0.7, f"R_A = {RA:,.0f} N", ha="left", fontsize=8)
    ax0.text(L, -0.7, f"R_B = {RB:,.0f} N", ha="right", fontsize=8)
    ax0.set_ylim(-0.9, 1.55)
    ax0.set_yticks([])
    ax0.set_title("Beam model")

    axV.plot(x, V, color="tab:blue")
    axV.fill_between(x, V, 0, alpha=0.2)
    axV.axhline(0, color="black", lw=0.8)
    axV.set_ylabel("V [N]")
    axV.set_title("Shear force diagram (SFD)")

    axM.plot(x, M, color="tab:red")
    axM.fill_between(x, M, 0, color="tab:red", alpha=0.2)
    axM.axhline(0, color="black", lw=0.8)
    axM.set_ylabel("M [N·m]")
    axM.set_title("Bending moment diagram (BMD)")
    axM.plot(xm, Mmax, "ko")
    axM.annotate(f"Mmax = {Mmax:,.0f} N·m\n(x = {xm:.2f} m)", (xm, Mmax),
                 textcoords="offset points", xytext=(10, -34))
    axM.set_xlabel("x [m]")
    axM.set_xlim(-0.05 * L, 1.05 * L)
    for ax in (axV, axM):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


tab_sim, tab_data = st.tabs(["시뮬레이터", "데이터 분석"])

with tab_sim:
    col_in, col_out = st.columns([3, 2])

    with col_in:
        with st.expander("문장으로 입력하기 (Claude)"):
            st.text_area("교재 문제 문장", key="problem_text",
                         placeholder="길이 6 m 단순보의 왼쪽 지점에서 2 m 떨어진 곳에 10 kN 집중하중이 작용한다.")
            st.button("슬라이더에 적용", on_click=apply_text)
            for m in st.session_state.get("parse_msgs", []):
                if m.startswith("문장의 조건"):
                    st.info(m)
                else:
                    st.warning(m)

        L = st.slider("보의 길이 L [m]", 1.0, 10.0, step=0.05, key="L")
        rho = st.slider("질량밀도 ρ [kg/m³]  (0 = 자중 무시, 목재≈500, 콘크리트≈2400, 강재≈7850)",
                        0, 8000, step=10, key="rho")
        b_cm = st.slider("단면 폭 b [cm]", 5, 50, step=1, key="b_cm")
        h_cm = st.slider("단면 높이 h [cm]", 5, 100, step=1, key="h_cm")
        # a 슬라이더의 최댓값(L)이 바뀌면 Streamlit이 a를 0으로 되돌리므로,
        # 마지막 a를 따로 보관했다가 매번 다시 넣는다. 작용점은 L을 넘을 수 없다.
        st.session_state["a"] = min(st.session_state["a_keep"], L)
        a = st.slider("집중하중 작용점 a [m] (왼쪽 지점 기준, 최대 L)", 0.0, L, step=0.05, key="a",
                      on_change=lambda: st.session_state.update(a_keep=st.session_state["a"]))
        st.session_state["a_keep"] = a
        P = st.slider("집중하중 크기 P [N]", 0, 50000, step=10, key="P")

        b, h = b_cm / 100, h_cm / 100
        r = solve_beam(L, rho, b, h, a, P)
        fig = draw(L, a, P, r)
        st.pyplot(fig)
        plt.close(fig)

    with col_out:
        RA, RB, w, xm, Mmax = r["R_A_N"], r["R_B_N"], r["w_N_m"], r["x_Mmax_m"], r["M_max_Nm"]
        c1, c2 = st.columns(2)
        c1.metric("R_A (왼쪽 핀)", f"{RA:,.1f} N")
        c2.metric("R_B (오른쪽 롤러)", f"{RB:,.1f} N")
        c1.metric("Mmax", f"{Mmax:,.1f} N·m")
        c2.metric("Mmax 위치 x", f"{xm:.3f} m")
        st.caption(f"자중 w = ρ·b·h·g = {w:,.1f} N/m · 검산: R_A + R_B = {RA + RB:,.1f} N = P + wL")
        if abs(xm - a) > 1e-9:
            st.info(f"자중 때문에 Mmax 위치가 하중점(a = {a:.2f} m)에서 {xm - a:+.3f} m 옮겨졌습니다.")

        st.markdown("**현재 값을 대입한 식**")
        st.latex(rf"0 \le x < {a:.2f}:\quad V = {RA:,.1f} - {w:,.1f}x")
        st.latex(rf"\qquad M = {RA:,.1f}x - {w / 2:,.1f}x^2")
        st.latex(rf"{a:.2f} < x \le {L:.2f}:\quad V = {RA - P:,.1f} - {w:,.1f}x")
        st.latex(rf"\qquad M = {RA - P:,.1f}x - {w / 2:,.1f}x^2 + {P * a:,.1f}")

        st.markdown("**AI 해설**")
        payload = result_summary(r, L, rho, b, h, a, P)
        if st.button("Claude 해설 보기"):
            from ai import explain_result
            st.session_state["explain"] = (payload, *explain_result(payload))
        ex = st.session_state.get("explain")
        if ex and ex[0] == payload:          # 값이 바뀌면 지난 해설은 숨긴다
            _, text, warn = ex
            if text:
                st.success(text)
            if warn:
                st.warning(warn)

with tab_data:
    st.subheader("시뮬레이션 데이터셋 분석 (1,512행)")
    st.caption("L 2~8 m × a/L 0.1~0.9 × P 5~30 kN × 재료 4종, 단면 20×40 cm 고정. 계산 엔진(beam.py)으로 생성")
    df = make_sweep()
    st.download_button("sweep.csv 내려받기", df.to_csv(index=False).encode("utf-8-sig"),
                       "sweep.csv", "text/csv")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**하중 위치(a/L)별 평균 Mmax** — 재료별")
        st.line_chart(df.pivot_table(index="a_ratio", columns="material", values="M_max_Nm", aggfunc="mean"))
    with c2:
        st.markdown("**재료별 Mmax 중 자중 몫 (평균)**")
        st.bar_chart(df.groupby("material")["selfweight_share"].mean())
    st.markdown("**기술통계 (Mmax, N·m)**")
    st.dataframe(df["M_max_Nm"].describe().round(1).to_frame().T)
    st.markdown("**Pearson 상관 (Mmax와)**")
    st.dataframe(df[["L_m", "a_ratio", "P_N", "rho", "M_max_Nm"]].corr()[["M_max_Nm"]].round(3).T)
    st.caption("a/L의 상관이 0인 것은 Mmax가 a/L = 0.5를 중심으로 대칭인 포물선이기 때문이다.")
