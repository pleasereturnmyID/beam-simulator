"""ai.py - Gemini API 연동 (Google AI Studio 키)
(A) 교재 문장 → 입력 JSON 파서   (B) 계산 결과 JSON → 해설 튜터
Gemini는 계산하지 않는다. 숫자 계산은 beam.py가 한다.
실패(키 없음, 네트워크, 모델명 오류, JSON 오류)는 모두 AIError로 바꾸고,
앱에는 (값 또는 None, 메시지)만 돌려주므로 계산 화면은 멈추지 않는다.
"""
import json
import re

import streamlit as st

DEFAULT_MODEL = "gemini-3-flash-preview"   # 무료 등급이 있는 Flash 모델. Secrets의 GEMINI_MODEL로 변경 가능

# 슬라이더 범위와 같게 맞춘 검증 범위
LIMITS = {"L_m": (1.0, 10.0), "a_m": (0.0, 10.0), "P_N": (0.0, 50000.0),
          "rho_kg_m3": (0.0, 8000.0), "b_m": (0.05, 0.5), "h_m": (0.05, 1.0)}
DEFAULTS = {"rho_kg_m3": 0.0, "b_m": 0.2, "h_m": 0.4}

PARSER_PROMPT = """[역할] 너는 정역학 문제 문장에서 수치 조건만 뽑아내는 파서다.
[조건]
1. 지원 범위: 왼쪽 핀·오른쪽 롤러 단순보, 아래 방향 집중하중 1개. 캔틸레버, 양단 고정보, 하중 2개 이상, 자중 외 분포하중이면 supported=false.
2. 단위 환산만 한다: 길이 m, 하중 N, 밀도 kg/m^3. 1 kN = 1000 N, 1 kgf = 9.81 N, 1 tf = 9810 N, 1 cm = 0.01 m, 1 mm = 0.001 m.
3. a는 왼쪽 지점에서 잰 거리다. "오른쪽 지점에서 d"이면 a = L - d, "중앙"이면 a = L/2.
4. 반력·전단력·모멘트는 계산하지 않는다.
5. 문장에 없는 값은 추측하지 말고 null로 둔다.
6. 바꾼 내용이나 모호한 점은 note에 한국어 한 문장으로 적는다.
[출력 형식] 다른 말 없이 아래 키만 가진 JSON 하나만 출력한다.
{"supported": true, "L_m": null, "a_m": null, "P_N": null, "rho_kg_m3": null, "b_m": null, "h_m": null, "note": ""}"""

TUTOR_PROMPT = """[역할] 너는 건축공학과 1학년에게 정역학을 가르치는 조교다.
[조건]
1. 입력 JSON에 있는 숫자만 인용한다. 새로 계산하거나 숫자를 바꾸지 않는다. N, N·m 값을 1000으로 나눠 kN, kN·m로 적는 것은 허용한다.
2. dM/dx = V 관계로 Mmax 위치를 설명한다. mmax_at_load_point가 false이면 자중 때문에 위치가 하중점에서 벗어났다고 설명한다.
3. 한국어, 5문장 이내, 존댓말.
[출력 형식]
① 한 줄 요약
② 반력 해석 (어느 지점이 더 많이 받치는지와 이유)
③ Mmax 위치와 이유
④ 설계 시 주의점 한 가지"""


class AIError(Exception):
    pass


def _secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:          # secrets.toml이 없을 때
        return default


@st.cache_resource
def _client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def has_key():
    return bool(_secret("GEMINI_API_KEY"))


def ask_gemini(system_prompt, user_text, temperature=0.2, json_mode=False):
    """두 프롬프트가 함께 쓰는 공통 함수. 실패는 모두 AIError."""
    key = _secret("GEMINI_API_KEY")
    if not key:
        raise AIError("GEMINI_API_KEY가 설정되지 않았습니다 (Streamlit Secrets 또는 .streamlit/secrets.toml).")
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain")
        res = _client(key).models.generate_content(
            model=_secret("GEMINI_MODEL", DEFAULT_MODEL), contents=user_text, config=cfg)
        text = (res.text or "").strip()
    except Exception as e:
        raise AIError(f"Gemini 호출 실패 ({type(e).__name__}): {e}") from e
    if not text:
        raise AIError("Gemini 응답이 비어 있습니다.")
    return text


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise AIError("Gemini 응답이 JSON 형식이 아닙니다.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise AIError("Gemini 응답이 JSON 형식이 아닙니다.") from e


def validate_params(p):
    """AI 출력을 그대로 믿지 않고 다시 검사한다. → (params 또는 None, 메시지 목록)"""
    msgs = []
    if not p.get("supported", False):
        return None, ["지원하지 않는 문제입니다 (단순보 + 집중하중 1개만 지원). " + (p.get("note") or "")]
    for k in ("L_m", "a_m", "P_N"):
        if not isinstance(p.get(k), (int, float)):
            return None, [f"문장에서 {k}를 찾지 못했습니다. 슬라이더 값은 바뀌지 않았습니다."]
    out = {}
    for k, (lo, hi) in LIMITS.items():
        v = p.get(k)
        if v is None:
            v = DEFAULTS[k]
            msgs.append(f"{k} 값이 문장에 없어 기본값 {v}을(를) 씁니다.")
        v = float(v)
        if not lo <= v <= hi:
            return None, [f"{k} = {v:g} 이(가) 허용 범위 {lo:g}~{hi:g} 밖입니다. 슬라이더 값은 바뀌지 않았습니다."]
        out[k] = v
    if out["a_m"] > out["L_m"]:
        return None, [f"작용점 a = {out['a_m']:g} m가 보 길이 L = {out['L_m']:g} m를 넘습니다."]
    if p.get("note"):
        msgs.append("Gemini 메모: " + p["note"])
    return out, msgs


def parse_problem(text):
    """프롬프트 A. → (params 또는 None, 메시지 목록). 예외를 밖으로 내보내지 않는다."""
    if not text or not text.strip():
        return None, ["문장을 입력해 주세요."]
    try:
        raw = _extract_json(ask_gemini(PARSER_PROMPT, text, temperature=0.0, json_mode=True))
    except AIError as e:
        return None, [f"{e} 슬라이더 값은 바뀌지 않았습니다."]
    return validate_params(raw)


def unknown_numbers(text, payload):
    """해설 속 숫자 중 결과 값(또는 그 값/1000)을 같은 자리수로 반올림해도 나오지 않는 숫자.
    10 미만 정수(①~④ 번호, '2개' 등)는 검사하지 않는다."""
    vals = [float(v) for v in payload.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    cands = [c for v in vals for c in (v, v / 1000)]
    bad = []
    for s in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        s = s.replace(",", "")
        if s.isdigit() and int(s) < 10:
            continue
        d = len(s.split(".")[1]) if "." in s else 0
        if not any(abs(round(c, d) - float(s)) < 1e-9 for c in cands):
            bad.append(s)
    return bad


def explain_result(payload):
    """프롬프트 B. → (해설 또는 None, 경고 또는 None)"""
    try:
        text = ask_gemini(TUTOR_PROMPT, json.dumps(payload, ensure_ascii=False))
    except AIError as e:
        return None, f"{e} 계산 결과와 그래프는 그대로 유효합니다."
    bad = unknown_numbers(text, payload)
    warn = f"해설에 계산 결과에 없는 숫자가 있습니다: {', '.join(bad)}. 위 계산값을 기준으로 보세요." if bad else None
    return text, warn
