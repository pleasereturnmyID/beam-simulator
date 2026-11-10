# 집중하중이 가해지는 단순보의 내부력 시뮬레이터

인공지능과 디지털 사고 팀프로젝트 (서준석, 오채윤, 우한결)

슬라이더로 단순보 조건(L, ρ, b, h, a, P)을 바꾸면 반력, V(x)·M(x) 식, 보 모형과 SFD·BMD, Mmax를 실시간으로 보여 주는 Streamlit 앱입니다. 계산은 파이썬(`beam.py`)이 하고, Claude는 교재 문장을 입력값으로 바꾸는 일과 결과 해설만 맡습니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `beam.py` | 계산 엔진: 자중, 반력, V(x), M(x), Mmax 위치 탐색 |
| `app.py` | Streamlit 화면: 슬라이더, 보 모형·SFD·BMD, 식, Claude 버튼, 데이터 분석 탭 |
| `ai.py` | Claude 연동: 프롬프트 A(문장 파서), 프롬프트 B(해설 튜터), AI 출력 검증 |
| `sweep.py` | 1,512행 시뮬레이션 데이터셋 생성과 통계·회귀 분석 |
| `analysis_colab.ipynb` | Google Colab용 분석 노트북 |
| `tests/` | 검산 테스트 5개 + 무작위 평형 검사 + 화면 동작 테스트 |

## 실행 방법 (내 컴퓨터)

```bash
pip install -r requirements.txt
python -m pytest -q          # 31 passed 확인
streamlit run app.py         # 브라우저에서 http://localhost:8501
```

Claude 기능(문장 입력, 해설)을 쓰려면 `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사하고 Anthropic Console에서 발급한 API 키를 넣습니다. 키가 없어도 계산과 그래프는 그대로 동작하고, Claude 버튼을 누르면 경고만 뜹니다.

## Google Colab에서 데이터 분석

1. Colab에서 `analysis_colab.ipynb`를 엽니다.
2. 왼쪽 파일 패널에 `beam.py`, `sweep.py`를 업로드합니다.
3. 위에서부터 셀을 실행하면 기준 예제 검산, `sweep.csv` 생성, 통계·회귀 결과와 그래프 3개가 나옵니다.

## 배포 (Streamlit Community Cloud)

1. 이 폴더를 GitHub 저장소에 올립니다 (`secrets.toml`은 `.gitignore`로 제외됨).
2. share.streamlit.io에서 저장소와 `app.py`를 고릅니다.
3. Advanced settings → Secrets에 `ANTHROPIC_API_KEY = "..."` 한 줄을 넣고 배포합니다.

## 기준 예제 (검산 완료)

L = 6 m, ρ = 7,850 kg/m³, b = 0.20 m, h = 0.40 m, a = 2 m, P = 10,000 N
→ R_A = 25,148.71 N, R_B = 21,815.37 N, Mmax = 38,624.84 N·m (x = 2.459 m)
