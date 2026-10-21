"""Generate the notebooks of the dam-inflow model comparison study.

* ``01_LSTM.ipynb`` ... ``10_Exformer.ipynb``  one notebook per model (methodology + training of the three input sets +
  evaluation + interpretation), all built from the same template so that the paper's methodology section can be
  written consistently,
* ``11_Comparison_Overall.ipynb``               comparison of all models over the full train / validation / test periods,
* ``12_Comparison_Peaks.ipynb``                 comparison of the flood-peak reproduction (event-based metrics).

Re-run ``python make_notebooks.py`` after changing the templates, then execute the notebooks with
``python run_notebooks.py`` (or in Jupyter with the ``nh`` kernel).
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from make_configs import MODEL_ORDER, MODEL_SPECS
from model_docs import MODEL_DOCS, MODEL_FAMILY, methodology_markdown
from result_notes import COMPARISON_NOTES, RESULT_NOTES

HERE = Path(__file__).resolve().parent
KERNEL = {"display_name": "nh", "language": "python", "name": "nh"}
NB_INDEX = {m: i + 1 for i, m in enumerate(MODEL_ORDER)}
N_MODELS = len(MODEL_ORDER)
NB_OVERALL = f"{N_MODELS + 1:02d}_Comparison_Overall.ipynb"
NB_PEAKS = f"{N_MODELS + 2:02d}_Comparison_Peaks.ipynb"


def nb_filename(model: str) -> str:
    return f"{NB_INDEX[model]:02d}_{MODEL_SPECS[model]['name'].replace('-', '')}.ipynb"


# ====================================================================================================================
# shared text blocks
# ====================================================================================================================
STUDY_TABLE = """
| 항목 | 내용 |
|---|---|
| 대상 | 용담댐 (유역면적 930.42 km²) 시간 유입량 (m³/s) |
| 기간 | 2006-01-01 ~ 2025-12-31 (20년, 175,320 시간) |
| 학습 / 검증 / 테스트 | 2006–2017 (12년) / 2018–2021 (4년) / 2022–2025 (4년) |
| 입력세트 A | 유역평균 강수량 `rf_avg` (1변수) |
| 입력세트 B | 관측소별 강수량 `rf_<관측소>` × 7 (7변수) |
| 입력세트 C | A + B (8변수) |
| 해상도 | 시간(hourly). MTS-LSTM만 일(daily) 강우를 추가로 사용 |
| 공통 학습 설정 | Adam, lr 1e-3, batch 256, MSE, seq_length 720 h, epoch 30–50 (early stopping patience 5), seed 111 |
| 평가 epoch | 검증 loss가 최소인 epoch (early stopping은 학습을 멈출 뿐 가중치를 되돌리지 않음) |
| 데이터셋 클래스 | `hourlycsv` (`data/YDDam_paper/`) — `prepare_data.py` 로 생성 |
| 설정 파일 | `configs/<model>_<set>.yml` — `make_configs.py` 로 생성 |
| 공통 코드 | `common.py` (run 관리·평가·그림), `model_docs.py` (모델 설명) |
"""

SETUP_CODE = '''%matplotlib inline
import sys, warnings
warnings.filterwarnings("ignore", message=".*IProgress.*")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

sys.path.insert(0, str(Path.cwd()))
import common
from common import SETS, MODEL_NAME, MODEL_COLOR, SET_COLOR, SET_STYLE, PERIODS, PERIOD_LABEL
from model_docs import MODEL_DOCS, MODEL_FAMILY

warnings.filterwarnings("ignore", category=FutureWarning)
common.set_plot_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
'''

DATA_OVERVIEW_CODE = '''rain = common.load_rainfall(["rf_avg"])["rf_avg"]
inflow = common.load_observed_inflow()
annual = pd.DataFrame({"rain_mm": rain.resample("YE").sum(), "inflow_mean": inflow.resample("YE").mean(),
                       "inflow_max": inflow.resample("YE").max()})
annual.index = annual.index.year
annual = annual.loc[2006:2025]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
axes[0].bar(annual.index, annual["rain_mm"], color="#4c72b0")
axes[0].set_ylabel("Annual rainfall (mm)")
axes[1].bar(annual.index, annual["inflow_mean"], color="#55a868")
axes[1].set_ylabel("Mean inflow (m³/s)")
axes[2].bar(annual.index, annual["inflow_max"], color="#c44e52")
axes[2].set_ylabel("Max hourly inflow (m³/s)")
for ax in axes:
    ax.axvspan(2005.5, 2017.5, color="gray", alpha=0.08)
    ax.axvspan(2017.5, 2021.5, color="orange", alpha=0.10)
    ax.axvspan(2021.5, 2025.5, color="green", alpha=0.10)
axes[0].text(2011.5, axes[0].get_ylim()[1] * 0.9, "train", ha="center", fontsize=14)
axes[0].text(2019.5, axes[0].get_ylim()[1] * 0.9, "validation", ha="center", fontsize=14)
axes[0].text(2023.5, axes[0].get_ylim()[1] * 0.9, "test", ha="center", fontsize=14)
axes[2].set_xlabel("Year")
axes[2].set_xticks(annual.index)
axes[2].tick_params(axis="x", rotation=45)
fig.suptitle("Yongdam Dam: annual basin-average rainfall, mean and maximum hourly inflow (2006-2025)", fontsize=16)
fig.tight_layout()
plt.show()
'''


# ====================================================================================================================
# per-model notebook
# ====================================================================================================================
def model_notebook(model: str) -> nbformat.NotebookNode:
    name = MODEL_SPECS[model]["name"]
    nh_model = MODEL_SPECS[model]["nh_model"]
    idx = NB_INDEX[model]
    cells = []

    cells.append(new_markdown_cell(f"""# {idx:02d}. {name} — 용담댐 시간 유입량 예측 (입력세트 A / B / C)

**모델**: {name} (`model: {nh_model}`) — {MODEL_FAMILY[model]}
**한 줄 요약**: {MODEL_DOCS[model]['summary']}

이 노트북은 모델 비교 연구({N_MODELS}개 모델 × 3개 입력세트)의 **{name} 부분**입니다. 모든 모델 노트북이 같은 구성을 가집니다.

1. 모델 설명 (논문 methodology용)
2. 자료와 실험 설정
3. 학습 (입력세트 A, B, C) 과 학습곡선
4. 학습 / 검증 / 테스트 기간 성능
5. 수문곡선 · 산점도 · 유량지속곡선
6. 홍수 첨두 재현
7. 결과 해석
8. 요약 (비교 노트북 `{NB_OVERALL}`, `{NB_PEAKS}` 에서 사용하는 `results/{model}_<set>/` 생성)

**연구 설정 요약**
{STUDY_TABLE}
"""))

    cells.append(new_markdown_cell(methodology_markdown(model)))

    cells.append(new_markdown_cell("""## 2. 자료와 실험 설정

`common.py` 가 run 디렉터리 탐색, 평가, 그림을 모두 담당하므로 노트북 코드는 짧습니다. 아래 셀은 공통 설정을 불러오고 그림 스타일(큰 글꼴,
300 dpi 저장)을 지정합니다."""))
    cells.append(new_code_cell(SETUP_CODE + f'\nMODEL = "{model}"\nNAME = MODEL_NAME[MODEL]\nprint(NAME, "->", common.MODEL_SPECS[MODEL]["nh_model"])'))

    cells.append(new_markdown_cell("""### 2.1 자료 개요

연도별 유역평균 강수량과 유입량(평균·최대)입니다. 음영은 학습(회색) / 검증(주황) / 테스트(녹색) 기간입니다.
테스트 기간(2022–2025)에는 2023년과 2024년의 큰 홍수가 포함되어 있고, 학습 기간에는 2020년(최대 4,395 m³/s)이 **포함되지 않는다**는 점에 유의합니다
(2020년은 검증 기간). 즉 모델은 학습에서 본 적 없는 크기의 홍수를 검증·테스트에서 재현해야 하며, 이것이 모델 간 외삽 능력 차이를 드러냅니다."""))
    cells.append(new_code_cell(DATA_OVERVIEW_CODE))

    cells.append(new_markdown_cell(f"""### 2.2 입력세트별 설정 파일

세 입력세트의 설정은 `dynamic_inputs` (와 모델에 따라 그에 연동된 몇 개 키)만 다릅니다. 아래 표는 공통 설정과 다른 항목만 보여 줍니다.
전체 파일은 `configs/{model}_<set>.yml` 입니다."""))
    cells.append(new_code_cell('''rows = {}
for s in SETS:
    rows[f"Set {s}"] = {k: str(v) for k, v in common.model_specific_config(MODEL, s).items()}
cfg_table = pd.DataFrame(rows).fillna("")
display(cfg_table)
print(Path(common.config_path(MODEL, "A")).read_text())'''))

    cells.append(new_markdown_cell(f"""## 3. 학습

`common.train()` 은 완료된 run (`runs/{model}_<set>_<날짜>_<시각>/`) 이 있으면 학습을 건너뛰고, 없으면 `nh-run train` 을 서브프로세스로
실행합니다 (`train_all.py` 로 미리 학습해 두어도 같은 run 을 사용합니다). 이어서 `common.evaluate_run()` 이

1. `output.log` 에서 epoch별 학습/검증 loss 와 검증 NSE/KGE 를 읽고,
2. 검증 loss 최소 epoch 을 골라 학습·검증·테스트 기간을 평가(`nh-run evaluate`)하고,
3. 시간 단위 예측을 m³/s 로 통일해 지표를 계산한 뒤 `results/{model}_<set>/` 에 저장합니다."""))
    cells.append(new_code_cell('''summaries, histories, preds = {}, {}, {}
for s in SETS:
    run_dir = common.train(MODEL, s, in_process=False)
    summaries[s] = common.evaluate_run(MODEL, s, run_dir)
    histories[s] = common.load_saved_history(MODEL, s)
    preds[s] = common.load_saved_predictions(MODEL, s)
    print(f"Set {s}: run={summaries[s]['run_dir']}  epochs={summaries[s]['epochs_trained']}  best={summaries[s]['best_epoch']}  "
          f"params={summaries[s]['n_parameters']:,}  {summaries[s]['epoch_time_mean_s']:.0f} s/epoch")'''))

    cells.append(new_markdown_cell("""### 3.1 학습곡선

왼쪽: 학습 loss(표준화된 MSE), 가운데: 검증 loss, 오른쪽: 검증 기간(2018–2021) NSE. 점선은 검증 loss 최소 epoch(평가에 쓴 체크포인트)입니다.
early stopping 은 30 epoch 이후 검증 loss 가 5 epoch 연속 개선되지 않으면 학습을 멈춥니다.

- 학습 loss 가 계속 내려가는데 검증 loss 가 올라가면 **과적합**, 둘 다 높은 채 머물면 **표현력 부족 또는 학습률 문제**입니다.
- 검증 NSE 가 들쭉날쭉하면 2020년 대홍수(검증 기간) 재현이 epoch 마다 흔들린다는 뜻입니다."""))
    cells.append(new_code_cell('''fig = common.plot_learning_curves(histories, title=f"{NAME}: learning curves by input set",
                                  labels={s: f"Set {s} ({common.INPUT_SET_NAMES[s]})" for s in SETS},
                                  colors=SET_COLOR, styles={s: "-" for s in SETS},
                                  best_epochs={s: summaries[s]["best_epoch"] for s in SETS})
common.save_fig(fig, f"{MODEL}_learning_curves")
plt.show()'''))

    cells.append(new_markdown_cell("""## 4. 학습 / 검증 / 테스트 기간 성능

모든 지표는 **시간 단위, m³/s** 로 계산합니다 (MC-LSTM 의 mm/h 예측도 변환). 지표의 의미:

| 지표 | 의미 | 이상값 |
|---|---|---|
| NSE | 관측 평균 대비 설명력. 제곱오차 기반이라 **홍수(고유량)** 에 민감 | 1 |
| KGE | 상관(r)·변동성비(α)·편향비(β)를 함께 보는 지표 | 1 |
| RMSE / MAE | 오차의 크기 (m³/s) | 0 |
| PBIAS | 총량 편향(%) (+ 과대, − 과소) | 0 |
| Alpha-NSE / Beta-NSE | 표준편차 비 / 평균 편향(관측 표준편차로 정규화) | 1 / 0 |
| FHV / FMS / FLV | 유량지속곡선의 고유량(상위 2%) / 중간 구간 기울기 / 저유량(하위 30%) 편향(%) | 0 |
| Peak-Timing | 관측 첨두(자동 탐지)의 시각 오차 평균(h) | 0 |
| Peak-MAPE | 자동 탐지된 모든 첨두의 평균 절대 백분율 오차(%) | 0 |
| Missed-Peaks | 자동 탐지 첨두 중 예측이 놓친 비율 | 0 |

(첨두 자동 탐지는 라이브러리 기본 규칙으로 작은 첨두까지 포함하므로 값이 큽니다. 큰 홍수사건에 대한 분석은 6절과 `" + NB_PEAKS + "` 참고.)"""))
    cells.append(new_code_cell('''rows = []
for s in SETS:
    for p in PERIODS:
        rows.append({"set": s, "period": p, **summaries[s]["metrics"][p]})
metrics_long = pd.DataFrame(rows).set_index(["set", "period"])[common.MAIN_METRICS]
display(metrics_long.round(3))
metrics_long.to_csv(common.RESULTS_DIR / f"{MODEL}_metrics_by_period.csv")'''))

    cells.append(new_markdown_cell("""### 4.1 기간별 NSE · KGE · PBIAS 비교

학습 기간 대비 검증·테스트 기간의 하락 폭이 **일반화 성능**을 보여 줍니다. 입력세트 간 차이는 관측소별 강우(B, C)가 유역평균(A)보다
추가 정보를 주는지를 말해 줍니다."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, metric in zip(axes, ["NSE", "KGE", "PBIAS"]):
    x = np.arange(len(PERIODS))
    w = 0.26
    for i, s in enumerate(SETS):
        vals = [summaries[s]["metrics"][p][metric] for p in PERIODS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=f"Set {s}", color=SET_COLOR[s], edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if metric != "PBIAS" else f"{v:+.0f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIODS])
    ax.set_title(metric)
    ax.axhline(common.METRIC_IDEAL[metric], color="black", ls="--", lw=1)
    if metric in ("NSE", "KGE"):
        ax.set_ylim(max(-0.5, min(0, ax.get_ylim()[0])), 1.05)
axes[0].legend(loc="lower left")
fig.suptitle(f"{NAME}: NSE / KGE / PBIAS by period and input set", fontsize=16)
fig.tight_layout()
common.save_fig(fig, f"{MODEL}_metrics_by_period")
plt.show()'''))

    cells.append(new_markdown_cell("""### 4.2 관측 vs 예측 산점도 (로그축)

행 = 기간, 열 = 입력세트. 로그축이므로 저유량(수 m³/s)과 홍수(수천 m³/s)의 거동을 한 그림에서 볼 수 있습니다.
1:1 선 아래로 휘면 고유량 과소, 왼쪽 위로 퍼지면 저유량 과대입니다. 관측값 0 (전체의 약 16%) 은 0.1 로 잘라 표시합니다."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for i, p in enumerate(PERIODS):
    for j, s in enumerate(SETS):
        df = preds[s][preds[s]["period"] == p]
        common.plot_scatter(df["obs"], df["sim"], title=f"Set {s} — {PERIOD_LABEL[p]}", ax=axes[i, j], color=SET_COLOR[s])
fig.suptitle(f"{NAME}: observed vs simulated hourly inflow", fontsize=17, y=1.0)
fig.tight_layout()
common.save_fig(fig, f"{MODEL}_scatter")
plt.show()'''))

    cells.append(new_markdown_cell("""## 5. 수문곡선 · 유량지속곡선

### 5.1 테스트 기간(2022–2025) 연도별 수문곡선

각 패널은 한 해의 시간 유입량입니다. 검은 선 = 관측, 색 선 = 입력세트별 예측, 위쪽 막대 = 유역평균 강우(역축).
홍수기(6–9월)의 첨두 크기와 상승·하강 시점, 건기의 저유량 수준을 봅니다."""))
    cells.append(new_code_cell('''test_years = range(2022, 2026)
fig, axes = plt.subplots(len(test_years), 1, figsize=(18, 4.2 * len(test_years)))
for ax, year in zip(axes, test_years):
    start, end = f"{year}-01-01", f"{year}-12-31 23:00"
    obs = preds["A"].loc[start:end, "obs"]
    common.plot_hydrograph(obs, {f"Set {s}": preds[s].loc[start:end, "sim"] for s in SETS}, rain=rain.loc[start:end],
                           title=f"{NAME} — test year {year}", ax=ax, colors={f"Set {s}": SET_COLOR[s] for s in SETS},
                           styles={f"Set {s}": "-" for s in SETS}, sim_lw=1.2, obs_lw=1.6)
fig.tight_layout()
common.save_fig(fig, f"{MODEL}_hydrograph_test")
plt.show()'''))

    cells.append(new_markdown_cell("""### 5.2 테스트 기간 최대 홍수사건 3개 확대

관측 첨두가 가장 큰 세 사건을 전후 5일 범위로 확대합니다. 시간 해상도이므로 첨두 **시각**까지 비교할 수 있습니다."""))
    cells.append(new_code_cell('''test_obs = preds["A"].loc[preds["A"]["period"] == "test", "obs"]
events = common.detect_events(test_obs, n_events=15)
top3 = sorted(events, key=lambda t: test_obs[t], reverse=True)[:3]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, peak in zip(axes, sorted(top3)):
    start, end = peak - pd.Timedelta(days=5), peak + pd.Timedelta(days=5)
    common.plot_hydrograph(preds["A"]["obs"], {f"Set {s}": preds[s]["sim"] for s in SETS}, rain=rain, start=start, end=end,
                           title=f"Peak {peak:%Y-%m-%d %H:00}  ({test_obs[peak]:,.0f} m³/s)", ax=ax,
                           colors={f"Set {s}": SET_COLOR[s] for s in SETS}, styles={f"Set {s}": "-" for s in SETS}, sim_lw=1.8)
    ax.tick_params(axis="x", rotation=30)
fig.suptitle(f"{NAME}: three largest flood events of the test period", fontsize=16)
fig.tight_layout()
common.save_fig(fig, f"{MODEL}_events_top3")
plt.show()'''))

    cells.append(new_markdown_cell("""### 5.3 유량지속곡선 (테스트 기간)

고유량(FHV, 왼쪽 음영)과 저유량(FLV, 오른쪽 음영) 구간에서 관측 곡선과의 간격이 각각 FHV, FLV 지표에 해당합니다."""))
    cells.append(new_code_cell('''fig, ax = plt.subplots(figsize=(10, 6.5))
common.plot_fdc(test_obs, {f"Set {s}": preds[s].loc[preds[s]["period"] == "test", "sim"] for s in SETS},
                title=f"{NAME}: flow duration curve, test period", colors={f"Set {s}": SET_COLOR[s] for s in SETS}, ax=ax)
common.save_fig(fig, f"{MODEL}_fdc_test")
plt.show()'''))

    cells.append(new_markdown_cell("""### 5.4 연도별 NSE

20년 전체를 연도별로 나눠 본 NSE 입니다. 배경 음영은 학습/검증/테스트 기간. 특정 해의 성능 급락은 보통 그 해의 큰 홍수를 놓쳤음을 뜻합니다."""))
    cells.append(new_code_cell('''yearly = {s: common.yearly_metrics(preds[s]) for s in SETS}
fig, ax = plt.subplots(figsize=(18, 5.5))
years = yearly["A"].index
w = 0.26
for i, s in enumerate(SETS):
    ax.bar(years + (i - 1) * w, yearly[s]["NSE"], w, label=f"Set {s}", color=SET_COLOR[s], edgecolor="black", lw=0.5)
ax.axvspan(2005.5, 2017.5, color="gray", alpha=0.08)
ax.axvspan(2017.5, 2021.5, color="orange", alpha=0.10)
ax.axvspan(2021.5, 2025.5, color="green", alpha=0.10)
ax.set_ylim(max(-1, min(0, np.nanmin([yearly[s]["NSE"].min() for s in SETS]) - 0.05)), 1.0)
ax.set_xticks(years)
ax.set_ylabel("NSE")
ax.set_xlabel("Year")
ax.set_title(f"{NAME}: yearly NSE (gray = train, orange = validation, green = test)")
ax.legend(loc="lower left")
common.save_fig(fig, f"{MODEL}_yearly_nse")
plt.show()
display(pd.concat({s: yearly[s][["NSE", "KGE"]] for s in SETS}, axis=1).round(3))'''))

    cells.append(new_markdown_cell("""## 6. 홍수 첨두 재현 (테스트 기간 상위 15개 사건)

관측 첨두를 서로 72시간 이상 떨어지도록 골라 독립 사건 15개를 정의하고, 사건마다

- **첨두 크기 오차** = (예측 첨두 − 관측 첨두) / 관측 첨두 × 100 (%) — 예측 첨두는 관측 첨두 ±24 h 안의 최대값
- **첨두 시각 오차** = 예측 첨두 시각 − 관측 첨두 시각 (h, + 는 지연)
- **사건 체적 오차** = 첨두 전 48 h ~ 후 96 h 구간 총량의 상대오차 (%)
- **사건 NSE** = 같은 구간의 NSE

를 계산합니다. 댐 운영에서는 평균적 적합도(NSE)보다 이 값들이 직접적인 관심사입니다."""))
    cells.append(new_code_cell('''event_tables = {s: common.load_saved_events(MODEL, s) for s in SETS}
best_set = max(SETS, key=lambda s: summaries[s]["metrics"]["test"]["NSE"])
print(f"테스트 NSE 가 가장 높은 입력세트: {best_set}")
t = event_tables[best_set].copy()
t[t.select_dtypes("number").columns] = t.select_dtypes("number").round(1)
display(t)
display(pd.DataFrame({s: summaries[s]["events_test"] for s in SETS}).round(2))'''))
    cells.append(new_code_cell('''fig, axes = plt.subplots(1, 2, figsize=(18, 6))
table = event_tables["A"]
x = np.arange(len(table))
w = 0.26
for i, s in enumerate(SETS):
    t = event_tables[s].reindex(table.index)
    axes[0].bar(x + (i - 1) * w, t["peak_error_pct"], w, label=f"Set {s}", color=SET_COLOR[s], edgecolor="black", lw=0.5)
    axes[1].bar(x + (i - 1) * w, t["timing_error_h"], w, label=f"Set {s}", color=SET_COLOR[s], edgecolor="black", lw=0.5)
labels = [f"{t:%Y-%m-%d}\\n{v:,.0f}" for t, v in zip(table.index, table["obs_peak"])]
for ax, ylabel, title in zip(axes, ["Peak magnitude error (%)", "Peak timing error (h)"],
                             ["Peak magnitude error (sim - obs) / obs", "Peak timing error (sim time - obs time)"]):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
axes[0].axhspan(-20, 20, color="green", alpha=0.08)
axes[1].axhspan(-3, 3, color="green", alpha=0.08)
axes[0].legend(loc="best")
fig.suptitle(f"{NAME}: peak errors of the 15 largest test events (x labels: date and observed peak in m³/s)", fontsize=15)
fig.tight_layout()
common.save_fig(fig, f"{MODEL}_peak_errors")
plt.show()'''))

    cells.append(new_markdown_cell("""## 7. 결과 해석

### 7.1 자동 요약 (지표 기반)

아래 글은 `common.narrative_for_run()` 이 지표 값으로부터 규칙적으로 생성한 요약입니다. 모든 모델 노트북에 같은 규칙을 적용하므로
모델 간 서술이 일관됩니다."""))
    cells.append(new_code_cell('''for s in SETS:
    display(Markdown(common.narrative_for_run(summaries[s])))'''))

    cells.append(new_markdown_cell(f"""### 7.2 {name} 의 결과 해석

{RESULT_NOTES.get(model, "_(학습이 끝난 뒤 작성)_").strip()}
"""))

    cells.append(new_markdown_cell(f"""## 8. 요약

아래 표가 이 노트북의 최종 결과이며, `results/{model}_<set>/summary.json` 으로 저장되어 비교 노트북에서 사용됩니다."""))
    cells.append(new_code_cell('''summary_table = pd.DataFrame({
    f"Set {s}": {
        "parameters": summaries[s]["n_parameters"],
        "epochs trained": summaries[s]["epochs_trained"],
        "best epoch": summaries[s]["best_epoch"],
        "early stopped": summaries[s]["early_stopped"],
        "s / epoch": round(summaries[s]["epoch_time_mean_s"], 1),
        "training time (min)": round(summaries[s]["train_time_total_s"] / 60, 1),
        "test NSE": round(summaries[s]["metrics"]["test"]["NSE"], 3),
        "test KGE": round(summaries[s]["metrics"]["test"]["KGE"], 3),
        "test RMSE (m³/s)": round(summaries[s]["metrics"]["test"]["RMSE"], 1),
        "test PBIAS (%)": round(summaries[s]["metrics"]["test"]["PBIAS"], 1),
        "test FHV (%)": round(summaries[s]["metrics"]["test"]["FHV"], 1),
        "peak |error| mean (%)": round(summaries[s]["events_test"]["peak_abs_error_mean_pct"], 1),
        "peak |timing| mean (h)": round(summaries[s]["events_test"]["timing_abs_error_mean_h"], 1),
    } for s in SETS})
display(summary_table)
summary_table.to_csv(common.RESULTS_DIR / f"{MODEL}_summary_table.csv")'''))

    nb = new_notebook(cells=cells, metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})
    return nb


# ====================================================================================================================
# comparison notebook 1: overall performance
# ====================================================================================================================
def comparison_overall_notebook() -> nbformat.NotebookNode:
    cells = []
    model_rows = "\n".join(f"| {NB_INDEX[m]:02d} | {MODEL_SPECS[m]['name']} | `{MODEL_SPECS[m]['nh_model']}` | {MODEL_FAMILY[m]} | "
                           f"{MODEL_DOCS[m]['summary']} | `{nb_filename(m)}` |" for m in MODEL_ORDER)
    cells.append(new_markdown_cell(f"""# {N_MODELS + 1}. 모델 비교 (1): 전체 기간 성능 — 학습 / 검증 / 테스트

{N_MODELS}개 딥러닝 모델 × 3개 입력세트(A: 유역평균 강우, B: 관측소별 강우 7개, C: A+B)로 용담댐 시간 유입량을 예측한 결과를
**전체 기간 지표** 로 비교합니다. 홍수 첨두 재현은 `{NB_PEAKS}` 에서 따로 다룹니다.

각 모델의 학습·평가는 모델별 노트북(`01_…` ~ `{N_MODELS:02d}_…`)에서 이루어지며, 이 노트북은 `results/<model>_<set>/` 에 저장된 결과만 읽습니다.

| # | 모델 | `model` 키 | 계열 | 한 줄 요약 | 노트북 |
|---|---|---|---|---|---|
{model_rows}

**연구 설정 요약**
{STUDY_TABLE}
"""))
    cells.append(new_code_cell(SETUP_CODE + '''
from make_configs import MODEL_ORDER
results = common.collect_results()
models = [m for m in MODEL_ORDER if m in set(results["model"])]
missing = [m for m in MODEL_ORDER if m not in models]
print(f"{len(models)} models with results; missing: {missing}")
rain = common.load_rainfall(["rf_avg"])["rf_avg"]
preds = {(m, s): common.load_saved_predictions(m, s) for m in models for s in SETS if common.load_summary(m, s)}
summaries = {(m, s): common.load_summary(m, s) for m in models for s in SETS if common.load_summary(m, s)}
# input set with the highest test NSE for every model -> used when one line per model is drawn
best_set = {m: max(SETS, key=lambda s: summaries[(m, s)]["metrics"]["test"]["NSE"]) for m in models}
label = {m: f"{MODEL_NAME[m]} ({best_set[m]})" for m in models}
best_set'''))

    cells.append(new_markdown_cell("""## 1. 모델 규모와 학습 비용

파라미터 수, 학습한 epoch 수, 최적 epoch, epoch당 시간, 총 학습 시간입니다. 같은 `hidden_size`(64)라도 구조에 따라 파라미터 수와
시간이 크게 다릅니다. (시간은 RTX 5090 Laptop GPU 기준, 서브프로세스 1개 또는 3개 동시 실행.)"""))
    cells.append(new_code_cell('''cost = results[results["period"] == "test"].pivot_table(
    index="model_name", columns="set", values=["n_parameters", "epochs_trained", "best_epoch", "epoch_time_mean_s", "train_time_total_s"],
    observed=True)
cost = cost.reindex(columns=pd.MultiIndex.from_product([["n_parameters", "epochs_trained", "best_epoch", "epoch_time_mean_s", "train_time_total_s"], SETS]))
cost["train_time_total_s"] = cost["train_time_total_s"] / 60
cost = cost.rename(columns={"train_time_total_s": "train_time_min"}, level=0)
display(cost.round(1))
cost.round(1).to_csv(common.RESULTS_DIR / "comparison_cost.csv")'''))
    cells.append(new_code_cell('''fig, axes = plt.subplots(1, 2, figsize=(18, 6))
common.plot_metric_bars(results, "epoch_time_mean_s", period="test", ax=axes[0], title="Mean training time per epoch (s)")
axes[0].set_ylabel("s / epoch")
axes[0].set_yscale("log")
common.plot_metric_bars(results, "n_parameters", period="test", ax=axes[1], title="Number of trainable parameters")
axes[1].set_yscale("log")
axes[1].set_ylabel("parameters")
fig.tight_layout()
common.save_fig(fig, "cmp_cost")
plt.show()'''))

    cells.append(new_markdown_cell("""## 2. 전체 기간 지표 (학습 / 검증 / 테스트)

행 = 모델, 열 = 지표 × 입력세트. 굵은 숫자 비교는 아래 히트맵이 더 편합니다.
모든 값은 시간 단위 m³/s 기준이며 `results/comparison_metrics_<period>.csv` 로 저장합니다."""))
    cells.append(new_code_cell('''tables = {}
for p in PERIODS:
    tables[p] = common.metric_table(results, p, metrics=("NSE", "KGE", "RMSE", "PBIAS", "FHV", "FLV"))
    tables[p].round(3).to_csv(common.RESULTS_DIR / f"comparison_metrics_{p}.csv")
    print(f"=== {PERIOD_LABEL[p]} ===")
    display(tables[p].round(3))'''))

    cells.append(new_markdown_cell("""### 2.1 NSE / KGE 히트맵 (모델 × 입력세트 × 기간)

초록일수록 좋습니다. 가로 방향으로 보면 입력세트의 효과, 세로 방향으로 보면 모델의 효과, 기간 블록을 비교하면 일반화(과적합) 정도를 알 수 있습니다."""))
    cells.append(new_code_cell('''def period_set_table(metric):
    t = results.pivot_table(index="model_name", columns=["period", "set"], values=metric, observed=True)
    return t.reindex(columns=pd.MultiIndex.from_product([list(PERIODS), SETS]))

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
for ax, metric in zip(axes, ["NSE", "KGE"]):
    t = period_set_table(metric)
    t.columns = [f"{p[:5]}\\n{s}" for p, s in t.columns]
    common.plot_heatmap(t, title=f"{metric} by model, period and input set", vmin=0.3, vmax=1.0, ax=ax, cbar_label=metric)
fig.tight_layout()
common.save_fig(fig, "cmp_heatmap_nse_kge")
plt.show()'''))

    cells.append(new_markdown_cell("""### 2.2 테스트 기간 막대그래프

NSE, KGE, RMSE, PBIAS 의 테스트 기간 값입니다. 점선은 이상값(1 또는 0)."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(2, 2, figsize=(20, 12))
for ax, metric in zip(axes.ravel(), ["NSE", "KGE", "RMSE", "PBIAS"]):
    common.plot_metric_bars(results, metric, period="test", ax=ax)
    if metric in ("NSE", "KGE"):
        ax.set_ylim(max(-0.2, ax.get_ylim()[0]), 1.05)
fig.tight_layout()
common.save_fig(fig, "cmp_bars_test")
plt.show()'''))

    cells.append(new_markdown_cell("""### 2.3 일반화: 학습 NSE 대 테스트 NSE

1:1 선에서 아래로 멀수록 학습 기간에 비해 테스트 기간 성능이 떨어진 것(과적합 또는 기간 간 수문특성 차이)입니다.
검증 NSE 를 x축으로 바꾸면 "검증으로 고른 모델이 테스트에서도 좋은가"를 볼 수 있습니다."""))
    cells.append(new_code_cell('''wide = results.pivot_table(index=["model", "model_name", "set"], columns="period", values="NSE", observed=True).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, xcol in zip(axes, ["train", "validation"]):
    for _, r in wide.iterrows():
        ax.scatter(r[xcol], r["test"], color=MODEL_COLOR[r["model"]], marker={"A": "o", "B": "s", "C": "^"}[r["set"]], s=90,
                   edgecolor="black", lw=0.5)
        ax.annotate(f"{r['model_name']}-{r['set']}", (r[xcol], r["test"]), fontsize=9, xytext=(4, 2), textcoords="offset points")
    lo = max(0, min(wide[xcol].min(), wide["test"].min()) - 0.05)
    ax.plot([lo, 1], [lo, 1], "k--", lw=1)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel(f"{xcol} NSE")
    ax.set_ylabel("test NSE")
    ax.set_title(f"test NSE vs {xcol} NSE (o = A, s = B, ^ = C)")
fig.tight_layout()
common.save_fig(fig, "cmp_generalization")
plt.show()'''))

    cells.append(new_markdown_cell("""## 3. 학습곡선 비교

각 모델의 **테스트 NSE 가 가장 높은 입력세트** 의 검증 NSE 추이입니다. 수렴 속도(몇 epoch 만에 안정되는가)와 안정성(흔들림)을 비교합니다.
MC-LSTM 은 손실이 mm/h 단위라 loss 축에서는 비교할 수 없으므로 검증 NSE 만 봅니다."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
for m in models:
    h = common.load_saved_history(m, best_set[m])
    axes[0].plot(h.index, h["NSE"], label=label[m], color=MODEL_COLOR[m], marker="o", ms=3)
    axes[1].plot(h.index, h["val_loss"] / h["val_loss"].iloc[0], label=label[m], color=MODEL_COLOR[m], marker="o", ms=3)
axes[0].set_ylabel("Validation NSE")
axes[0].set_ylim(0, 1)
axes[1].set_ylabel("Validation loss / loss at epoch 1")
axes[1].set_yscale("log")
for ax in axes:
    ax.set_xlabel("Epoch")
axes[0].legend(loc="lower right", ncol=2, fontsize=11)
fig.suptitle("Learning curves (best input set of every model)", fontsize=16)
fig.tight_layout()
common.save_fig(fig, "cmp_learning_curves")
plt.show()'''))

    cells.append(new_markdown_cell("""## 4. 테스트 기간 수문곡선 비교

연도별 패널에 모든 모델(각각 최적 입력세트)을 겹쳐 그립니다. 선이 많으므로 전체 모양(계절성, 홍수기 반응, 건기 수준)을 보는 용도이고,
사건별 세부 비교는 `" + NB_PEAKS + "` 에서 합니다."""))
    cells.append(new_code_cell('''obs_all = preds[(models[0], "A")]["obs"]
fig, axes = plt.subplots(4, 1, figsize=(20, 18))
for ax, year in zip(axes, range(2022, 2026)):
    start, end = f"{year}-01-01", f"{year}-12-31 23:00"
    common.plot_hydrograph(obs_all.loc[start:end], {label[m]: preds[(m, best_set[m])].loc[start:end, "sim"] for m in models},
                           rain=rain.loc[start:end], title=f"Test year {year} — all models (best input set each)", ax=ax,
                           colors={label[m]: MODEL_COLOR[m] for m in models}, sim_lw=1.0, obs_lw=2.0)
    ax.legend(loc="upper right", ncol=4, fontsize=10)
fig.tight_layout()
common.save_fig(fig, "cmp_hydrograph_test")
plt.show()'''))

    cells.append(new_markdown_cell("""### 4.1 유량지속곡선 비교 (테스트 기간)"""))
    cells.append(new_code_cell('''test_mask = preds[(models[0], "A")]["period"] == "test"
fig, ax = plt.subplots(figsize=(12, 7.5))
common.plot_fdc(obs_all[test_mask], {label[m]: preds[(m, best_set[m])].loc[test_mask, "sim"] for m in models},
                title="Flow duration curves, test period (best input set of every model)",
                colors={label[m]: MODEL_COLOR[m] for m in models}, ax=ax)
ax.legend(loc="lower left", ncol=2, fontsize=11)
common.save_fig(fig, "cmp_fdc_test")
plt.show()'''))

    cells.append(new_markdown_cell("""## 5. 연도별 · 월별 성능

### 5.1 연도별 NSE 히트맵 (각 모델의 최적 입력세트)

세로줄로 특정 해(예: 2020년 대홍수가 있는 검증 기간, 2023·2024년 테스트 기간)에 모든 모델이 함께 나빠지는지, 아니면 특정 모델만 나빠지는지
를 볼 수 있습니다."""))
    cells.append(new_code_cell('''yearly = pd.DataFrame({label[m]: common.yearly_metrics(preds[(m, best_set[m])])["NSE"] for m in models}).T
fig, ax = plt.subplots(figsize=(20, 8))
common.plot_heatmap(yearly, title="Yearly NSE (columns 2006-2017 train, 2018-2021 validation, 2022-2025 test)", vmin=0, vmax=1,
                    ax=ax, cbar_label="NSE")
common.save_fig(fig, "cmp_yearly_nse")
plt.show()'''))
    cells.append(new_markdown_cell("""### 5.2 월별 총량 편향 (테스트 기간)

월별 PBIAS 는 계절별 체계적 편향(예: 건기 과대, 홍수기 과소)을 보여 줍니다."""))
    cells.append(new_code_cell('''fig, ax = plt.subplots(figsize=(16, 6))
for m in models:
    df = preds[(m, best_set[m])]
    mm = common.monthly_metrics(df[df["period"] == "test"])
    ax.plot(mm.index, mm["PBIAS"], label=label[m], color=MODEL_COLOR[m], marker="o")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("PBIAS (%)")
ax.set_title("Monthly volume bias, test period (best input set of every model)")
ax.legend(ncol=4, fontsize=10)
common.save_fig(fig, "cmp_monthly_pbias")
plt.show()'''))

    cells.append(new_markdown_cell("""## 6. 입력세트의 효과

유역평균 강우(A) 대비 관측소별 강우(B)와 둘을 합친 경우(C)의 테스트 NSE 변화량입니다. 양수면 관측소별 강우가 도움이 된 것입니다.
7개 관측소 평균과 유역평균의 상관이 0.998 로 매우 높아 정보 이득은 **공간 분포**(어느 관측소에 비가 왔는가)에서만 나옵니다."""))
    cells.append(new_code_cell('''t = results[results["period"] == "test"].pivot_table(index="model_name", columns="set", values="NSE", observed=True)
delta = pd.DataFrame({"B − A": t["B"] - t["A"], "C − A": t["C"] - t["A"]})
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(delta))
ax.bar(x - 0.18, delta["B − A"], 0.36, label="Set B − Set A", color=SET_COLOR["B"], edgecolor="black", lw=0.5)
ax.bar(x + 0.18, delta["C − A"], 0.36, label="Set C − Set A", color=SET_COLOR["C"], edgecolor="black", lw=0.5)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(delta.index, rotation=30, ha="right")
ax.set_ylabel("Δ test NSE")
ax.set_title("Effect of station rainfall on test NSE (relative to basin-average rainfall)")
ax.legend()
common.save_fig(fig, "cmp_input_set_effect")
plt.show()
display(delta.round(3))'''))

    cells.append(new_markdown_cell("""## 7. 정확도 대 계산 비용

x축: epoch당 학습 시간(로그), y축: 테스트 NSE, 원 크기: 파라미터 수. 왼쪽 위가 "싸고 좋은" 모델입니다."""))
    cells.append(new_code_cell('''sub = results[results["period"] == "test"]
fig, ax = plt.subplots(figsize=(14, 8))
for _, r in sub.iterrows():
    ax.scatter(r["epoch_time_mean_s"], r["NSE"], s=30 + 400 * np.sqrt(r["n_parameters"] / sub["n_parameters"].max()),
               color=MODEL_COLOR[r["model"]], alpha=0.75, edgecolor="black", lw=0.5,
               marker={"A": "o", "B": "s", "C": "^"}[r["set"]])
    ax.annotate(f"{r['model_name']}-{r['set']}", (r["epoch_time_mean_s"], r["NSE"]), fontsize=9, xytext=(5, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Training time per epoch (s, log)")
ax.set_ylabel("Test NSE")
ax.set_ylim(max(0, sub["NSE"].min() - 0.05), 1)
ax.set_title("Accuracy vs. training cost (marker size ~ number of parameters; o = A, s = B, ^ = C)")
common.save_fig(fig, "cmp_accuracy_vs_cost")
plt.show()'''))

    cells.append(new_markdown_cell("""## 8. 종합 순위

테스트 기간의 NSE, KGE, RMSE, |PBIAS|, |FHV| 다섯 지표에 대한 순위의 평균입니다(각 모델의 최적 입력세트 기준과 세트별 기준 모두 제시).
순위 평균이 작을수록 종합적으로 좋습니다."""))
    cells.append(new_code_cell('''sub = results[results["period"] == "test"].copy()
sub["absPBIAS"] = sub["PBIAS"].abs()
sub["absFHV"] = sub["FHV"].abs()
rank_cols = {"NSE": False, "KGE": False, "RMSE": True, "absPBIAS": True, "absFHV": True}
ranks = pd.DataFrame({c: sub[c].rank(ascending=asc) for c, asc in rank_cols.items()})
sub["mean_rank"] = ranks.mean(axis=1)
ranking = sub.sort_values("mean_rank")[["model_name", "set", "NSE", "KGE", "RMSE", "PBIAS", "FHV", "mean_rank"]].reset_index(drop=True)
ranking.index += 1
display(ranking.round(3))
ranking.round(3).to_csv(common.RESULTS_DIR / "comparison_ranking_test.csv")

best_rows = sub.loc[sub.groupby("model", observed=True)["NSE"].idxmax()].sort_values("NSE", ascending=False)
display(best_rows[["model_name", "set", "NSE", "KGE", "RMSE", "PBIAS", "FHV", "FLV", "epoch_time_mean_s", "n_parameters"]].round(3).reset_index(drop=True))'''))

    cells.append(new_markdown_cell("""### 8.1 자동 생성 요약"""))
    cells.append(new_code_cell('''lines = []
top = best_rows.iloc[0]
lines.append(f"- 테스트 NSE 최고: **{top['model_name']} (Set {top['set']})** NSE {top['NSE']:.3f}, KGE {top['KGE']:.3f}, RMSE {top['RMSE']:.1f} m³/s.")
baseline = sub[(sub['model'] == 'lstm')].sort_values('NSE', ascending=False).iloc[0]
lines.append(f"- 기준 LSTM 최고 세트 (Set {baseline['set']}): NSE {baseline['NSE']:.3f}, KGE {baseline['KGE']:.3f}.")
better = best_rows[best_rows["NSE"] > baseline["NSE"] + 0.01]["model_name"].tolist()
worse = best_rows[best_rows["NSE"] < baseline["NSE"] - 0.01]["model_name"].tolist()
lines.append(f"- LSTM 보다 NSE 0.01 이상 높은 모델: {', '.join(str(x) for x in better) if better else '없음'}; 낮은 모델: {', '.join(str(x) for x in worse) if worse else '없음'}.")
pos = delta.mean()
lines.append(f"- 입력세트 효과(테스트 NSE 변화량의 모델 평균): B−A {pos['B − A']:+.3f}, C−A {pos['C − A']:+.3f}; "
             f"B 또는 C 가 A 보다 좋은 모델 수 {(delta.max(axis=1) > 0).sum()} / {len(delta)}.")
fast = sub.sort_values('epoch_time_mean_s').iloc[0]
slow = sub.sort_values('epoch_time_mean_s').iloc[-1]
lines.append(f"- epoch당 학습 시간: 최단 {fast['model_name']} {fast['epoch_time_mean_s']:.0f} s, 최장 {slow['model_name']} {slow['epoch_time_mean_s']:.0f} s "
             f"({slow['epoch_time_mean_s'] / fast['epoch_time_mean_s']:.0f}배).")
gap = (wide["train"] - wide["test"]).groupby(wide["model_name"], observed=True).mean().sort_values()
lines.append(f"- 학습−테스트 NSE 차이(세트 평균)가 가장 작은 모델: {gap.index[0]} ({gap.iloc[0]:+.3f}), 가장 큰 모델: {gap.index[-1]} ({gap.iloc[-1]:+.3f}).")
display(Markdown("\\n".join(lines)))'''))

    cells.append(new_markdown_cell(f"""## 9. 해석

{COMPARISON_NOTES.get("overall", "_(학습이 끝난 뒤 작성)_").strip()}
"""))

    cells.append(new_markdown_cell("""## 10. Future work: 하이퍼파라미터 튜닝

본 비교는 **모든 모델에 같은 학습 설정**(hidden 64, Adam 1e-3, batch 256, seq 720, epoch 30–50, early stopping) 을 적용한 것으로,
각 모델이 낼 수 있는 최고 성능이 아니라 "같은 예산에서의 기본 성능" 을 비교한 것입니다. 후속 연구에서 다룰 항목:

1. **모델별 탐색 공간**: 순환 모델은 `hidden_size`(32–256), `output_dropout`, `seq_length`(168–2160 h); 트랜스포머 계열은 층 수·헤드 수·`d_model`·
   위치 인코딩; Mamba 는 `d_state`·`expand`; xLSTM 은 블록 수와 sLSTM 위치; PeakFocus 는 첨두 라벨 임계와 손실 가중치; Exformer 는 패치 크기와
   극한 임계.
2. **학습률·배치·스케줄**: 학습률(1e-4 ~ 5e-3), 배치(64–512), cosine / OneCycle 스케줄, 학습률 워밍업(트랜스포머 계열에 특히 중요).
3. **손실 함수**: MSE 외에 NSE 손실(`NSE`), 첨두 가중 손실, 로그 변환 목표 — 저유량 과대·첨두 과소 편향에 직접 영향.
4. **입력 확장**: 기온·증발산·댐 방류·저수위 등 추가 변수, 정적 속성(다유역 학습 시).
5. **앙상블과 시드**: 시드 3–5개 평균(`nh-results-ensemble`)으로 모델 간 차이가 시드 변동보다 큰지 검정.
6. **탐색 방법**: Optuna / Ray Tune 등으로 검증 NSE 를 목적함수로 한 베이지안 탐색을 모델별 동일 시도 횟수(예: 30회)로 수행하고,
   튜닝 전후의 순위 변화를 보고.
"""))
    return new_notebook(cells=cells, metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})


# ====================================================================================================================
# comparison notebook 2: peaks
# ====================================================================================================================
def comparison_peaks_notebook() -> nbformat.NotebookNode:
    cells = []
    cells.append(new_markdown_cell(f"""# {N_MODELS + 2}. 모델 비교 (2): 홍수 첨두 재현 성능

댐 운영(홍수조절, 사전방류)에서는 평균적 적합도보다 **홍수 첨두의 크기와 시각, 사건 체적** 이 직접적인 관심사입니다.
이 노트북은 테스트 기간(2022–2025)의 주요 홍수사건에 대해 {N_MODELS}개 모델 × 3개 입력세트의 첨두 재현 성능을 비교합니다.

**연구 설정 요약**
{STUDY_TABLE}

## 1. 첨두 성능 지표 설명

### 1.1 사건(event) 정의

테스트 기간 관측 시간 유입량에서 값이 큰 순서로 시각을 고르되, 이미 고른 사건과 **72시간 이상** 떨어진 시각만 새 사건으로 인정하여
독립적인 홍수사건 **15개** 를 정의합니다(`common.detect_events`). 72시간은 이 유역의 홍수 지속시간(상승 1일 + 하강 2일 정도)을 고려한 값입니다.

### 1.2 사건별 지표 (`common.event_table`)

| 지표 | 정의 | 해석 |
|---|---|---|
| 관측 첨두 $Q_p^{{obs}}$ | 사건 시각의 관측 유입량 | — |
| 예측 첨두 $Q_p^{{sim}}$ | 관측 첨두 ±24 h 안의 예측 최대값 | 탐색창을 두어 시각이 약간 어긋나도 크기를 공정하게 비교 |
| **첨두 크기 오차** | $(Q_p^{{sim}} - Q_p^{{obs}}) / Q_p^{{obs}} \\times 100$ (%) | 0 이 이상. 음수 = 과소, ±20% 이내면 운영상 양호 |
| **첨두 시각 오차** | $t_p^{{sim}} - t_p^{{obs}}$ (h) | 0 이 이상. 양수 = 지연. ±3 h 이내면 양호 |
| **사건 체적 오차** | 첨두 −48 h ~ +96 h 구간 총량의 상대오차 (%) | 사전방류 판단에 쓰이는 홍수 총량 |
| **사건 NSE** | 같은 구간(6일)의 NSE | 사건 수문곡선 모양 전체의 적합도 |

사건별 값을 모델별로 집계한 요약 통계(`common.event_summary`): 첨두 크기 오차 평균·중앙값·평균 절대값, 시각 오차 평균·평균 절대값,
±3 h 이내 비율, ±20% 이내 비율, 체적 오차 평균, 사건 NSE 중앙값.

### 1.3 라이브러리 전체 기간 첨두 지표 (`neuralhydrology.evaluation.metrics`)

| 지표 | 정의 |
|---|---|
| **Peak-Timing** | 관측 시계열에서 `scipy.signal.find_peaks`(거리 100 h, prominence = 관측 표준편차)로 찾은 **모든** 첨두에 대해, ±3 h 탐색창 안 예측 최대값의 시각 차이 절대값 평균(h) |
| **Peak-MAPE** | 같은 첨두들에 대한 크기의 평균 절대 백분율 오차 (%) |
| **Missed-Peaks** | 예측 시계열에서 같은 방법으로 찾은 첨두와 ±3 h 안에서 짝이 맞지 않은 관측 첨두의 비율 |
| **FHV** | 유량지속곡선 상위 2% 구간의 체적 편향 (%) — 고유량 전반의 과대/과소 |

전체 기간 지표는 작은 첨두까지 모두 포함하므로 값이 크고, 사건별 지표는 **큰 홍수 15개** 에 초점을 둡니다. 두 관점을 함께 봅니다.
"""))
    cells.append(new_code_cell(SETUP_CODE + '''
from make_configs import MODEL_ORDER
results = common.collect_results()
models = [m for m in MODEL_ORDER if m in set(results["model"])]
rain = common.load_rainfall(["rf_avg"])["rf_avg"]
preds = {(m, s): common.load_saved_predictions(m, s) for m in models for s in SETS if common.load_summary(m, s)}
summaries = {(m, s): common.load_summary(m, s) for m in models for s in SETS if common.load_summary(m, s)}
best_set = {m: max(SETS, key=lambda s: summaries[(m, s)]["metrics"]["test"]["NSE"]) for m in models}
label = {m: f"{MODEL_NAME[m]} ({best_set[m]})" for m in models}
obs = preds[(models[0], "A")]
test_obs = obs.loc[obs["period"] == "test", "obs"]
events = common.detect_events(test_obs, n_events=15)
print(f"{len(models)} models, {len(events)} events")'''))

    cells.append(new_markdown_cell("""## 2. 테스트 기간 홍수사건

사건 목록(관측 첨두 크기, 선행 72 h 유역평균 강우량)과 테스트 기간 수문곡선 위의 사건 위치입니다."""))
    cells.append(new_code_cell('''event_info = pd.DataFrame({
    "obs_peak_m3s": [test_obs[t] for t in events],
    "rain_72h_before_mm": [rain.loc[t - pd.Timedelta(hours=72):t].sum() for t in events],
    "rain_max_1h_mm": [rain.loc[t - pd.Timedelta(hours=72):t].max() for t in events],
}, index=pd.Index(events, name="obs_peak_time"))
event_info["rank"] = event_info["obs_peak_m3s"].rank(ascending=False).astype(int)
display(event_info.round(1))
event_info.round(2).to_csv(common.RESULTS_DIR / "peak_events_test.csv")

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(test_obs.index, test_obs, color="black", lw=1.2, label="Observed")
ax.scatter(events, [test_obs[t] for t in events], color="red", s=70, zorder=5, label="Selected events (15)")
for t in events:
    ax.annotate(str(event_info.loc[t, "rank"]), (t, test_obs[t]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11, color="red")
ax.set_ylabel("Inflow (m³/s)")
ax.set_title("Test period (2022-2025): observed hourly inflow and the 15 selected flood events (number = rank by peak)")
ax.legend(loc="upper right")
common.save_fig(fig, "peaks_event_overview")
plt.show()'''))

    cells.append(new_markdown_cell("""## 3. 전체 기간 첨두 지표 (라이브러리 정의)

Peak-Timing, Peak-MAPE, Missed-Peaks, FHV 의 테스트 기간 값입니다. 모두 0 이 이상값입니다."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(2, 2, figsize=(20, 12))
for ax, metric in zip(axes.ravel(), ["Peak-Timing", "Peak-MAPE", "Missed-Peaks", "FHV"]):
    common.plot_metric_bars(results, metric, period="test", ax=ax)
fig.tight_layout()
common.save_fig(fig, "peaks_library_metrics")
plt.show()
lib = common.metric_table(results, "test", metrics=("Peak-Timing", "Peak-MAPE", "Missed-Peaks", "FHV"))
display(lib.round(2))
lib.round(3).to_csv(common.RESULTS_DIR / "peaks_library_metrics_test.csv")'''))

    cells.append(new_markdown_cell("""## 4. 사건별 첨두 크기 · 시각 오차

### 4.1 관측 첨두 대 예측 첨두 (각 모델의 최적 입력세트)

1:1 선에 가까울수록 좋습니다. 큰 사건에서 점이 선 아래로 처지면 **대홍수 과소** 경향입니다. 음영은 ±20% 구간."""))
    cells.append(new_code_cell('''event_tables = {(m, s): common.event_table(preds[(m, s)][preds[(m, s)]["period"] == "test"], events) for m in models for s in SETS}
fig, ax = plt.subplots(figsize=(11, 11))
for m in models:
    t = event_tables[(m, best_set[m])]
    ax.scatter(t["obs_peak"], t["sim_peak"], label=label[m], color=MODEL_COLOR[m], s=70, alpha=0.8, edgecolor="black", lw=0.5)
lim = max(test_obs.max(), max(t["sim_peak"].max() for t in event_tables.values())) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")
ax.fill_between([0, lim], [0, lim * 0.8], [0, lim * 1.2], color="green", alpha=0.08, label="±20%")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Observed peak (m³/s)")
ax.set_ylabel("Simulated peak (m³/s)")
ax.set_title("Observed vs simulated peak of the 15 test events")
ax.legend(fontsize=11, ncol=2, loc="upper left")
ax.set_aspect("equal")
common.save_fig(fig, "peaks_scatter")
plt.show()'''))

    cells.append(new_markdown_cell("""### 4.2 사건 × 모델 히트맵

첨두 크기 오차(%), 첨두 시각 오차(h), 사건 NSE 를 사건(행, 첨두 크기 순) × 모델(열) 로 봅니다. 행 방향으로 모든 모델이 같은 색이면
그 사건은 **입력(강우) 자체로 설명이 어려운 사건**(예: 댐 상류 지류의 국지 호우, 자료 오류)일 가능성이 큽니다."""))
    cells.append(new_code_cell('''order = event_info.sort_values("obs_peak_m3s", ascending=False).index
row_labels = [f"{t:%Y-%m-%d %Hh} ({event_info.loc[t, 'obs_peak_m3s']:,.0f})" for t in order]

def event_matrix(col):
    mat = pd.DataFrame({label[m]: event_tables[(m, best_set[m])].reindex(order)[col] for m in models})
    mat.index = row_labels
    return mat

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
common.plot_heatmap(event_matrix("peak_error_pct"), title="Peak magnitude error (%)", fmt="{:+.0f}", cmap="RdBu_r", vmin=-60, vmax=60, ax=axes[0], cbar_label="%")
common.plot_heatmap(event_matrix("timing_error_h"), title="Peak timing error (h)", fmt="{:+.0f}", cmap="RdBu_r", vmin=-12, vmax=12, ax=axes[1], cbar_label="h")
common.plot_heatmap(event_matrix("event_NSE"), title="Event NSE (-48 h ... +96 h)", fmt="{:.2f}", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[2], cbar_label="NSE")
for ax in axes[1:]:
    ax.set_yticklabels([])
fig.tight_layout()
common.save_fig(fig, "peaks_heatmaps")
plt.show()'''))

    cells.append(new_markdown_cell("""### 4.3 모델별 분포 (상자그림)

15개 사건에 대한 첨두 크기 오차와 시각 오차의 분포입니다. 상자가 0 에 가깝고 좁을수록 좋습니다. 각 모델의 최적 입력세트 기준."""))
    cells.append(new_code_cell('''fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, col, ylabel, band in zip(axes, ["peak_error_pct", "timing_error_h", "volume_error_pct"],
                                 ["Peak magnitude error (%)", "Peak timing error (h)", "Event volume error (%)"], [20, 3, 20]):
    data = [event_tables[(m, best_set[m])][col].dropna().values for m in models]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLOR[m])
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=1)
    ax.axhspan(-band, band, color="green", alpha=0.08)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([label[m] for m in models], rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " — 15 test events")
fig.tight_layout()
common.save_fig(fig, "peaks_boxplots")
plt.show()'''))

    cells.append(new_markdown_cell("""## 5. 주요 홍수사건 수문곡선 (상위 6개 사건, 모든 모델)

관측 첨두가 큰 순서로 6개 사건을 전후 4일 범위에서 모든 모델(최적 입력세트)과 함께 그립니다. 상승부 시작 시각, 첨두 크기·시각, 하강부 기울기를
모델별로 비교할 수 있습니다."""))
    cells.append(new_code_cell('''top6 = list(order[:6])
fig, axes = plt.subplots(3, 2, figsize=(24, 18))
for ax, peak in zip(axes.ravel(), top6):
    start, end = peak - pd.Timedelta(days=4), peak + pd.Timedelta(days=4)
    common.plot_hydrograph(obs["obs"], {label[m]: preds[(m, best_set[m])]["sim"] for m in models}, rain=rain, start=start, end=end,
                           title=f"Event {event_info.loc[peak, 'rank']}: {peak:%Y-%m-%d %H:00}, observed peak {test_obs[peak]:,.0f} m³/s",
                           ax=ax, colors={label[m]: MODEL_COLOR[m] for m in models}, sim_lw=1.4, obs_lw=2.2)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
common.save_fig(fig, "peaks_top6_hydrographs")
plt.show()'''))

    cells.append(new_markdown_cell("""## 6. 입력세트별 첨두 성능

관측소별 강우가 첨두 재현에 도움이 되는지: 모델 × 입력세트의 평균 절대 첨두 크기 오차와 평균 절대 시각 오차입니다."""))
    cells.append(new_code_cell('''rows = []
for (m, s), t in event_tables.items():
    rows.append({"model": m, "model_name": MODEL_NAME[m], "set": s, "period": "test", **{f"event_{k}": v for k, v in common.event_summary(t).items()}})
ev = pd.DataFrame(rows)
ev["model_name"] = pd.Categorical(ev["model_name"], categories=[MODEL_NAME[m] for m in models], ordered=True)
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
common.plot_metric_bars(ev, "event_peak_abs_error_mean_pct", ax=axes[0, 0], title="Mean |peak magnitude error| (%)")
common.plot_metric_bars(ev, "event_timing_abs_error_mean_h", ax=axes[0, 1], title="Mean |peak timing error| (h)")
common.plot_metric_bars(ev, "event_peak_error_mean_pct", ax=axes[1, 0], title="Mean peak magnitude error (%, sign = bias)")
axes[1, 0].axhline(0, color="black", lw=1)
common.plot_metric_bars(ev, "event_event_NSE_median", ax=axes[1, 1], title="Median event NSE")
axes[1, 1].set_ylim(max(-0.5, axes[1, 1].get_ylim()[0]), 1.05)
fig.tight_layout()
common.save_fig(fig, "peaks_by_input_set")
plt.show()
ev_table = ev.pivot_table(index="model_name", columns="set", observed=True,
                          values=["event_peak_abs_error_mean_pct", "event_timing_abs_error_mean_h", "event_peak_within_20pct_frac", "event_timing_within_3h_frac", "event_event_NSE_median"])
display(ev_table.round(2))
ev_table.round(3).to_csv(common.RESULTS_DIR / "peaks_event_summary_by_set.csv")'''))

    cells.append(new_markdown_cell("""## 7. 종합 순위 (첨두 관점)

사건별 지표 네 개(평균 |첨두 크기 오차|, 평균 |시각 오차|, 사건 NSE 중앙값, |평균 체적 오차|)와 라이브러리 지표 두 개(Peak-MAPE, |FHV|)에 대한
순위의 평균입니다. 모델 × 입력세트 {N_MODELS * 3}개 조합 모두를 순위 매깁니다."""))
    cells.append(new_code_cell('''lib_t = results[results["period"] == "test"][["model", "model_name", "set", "Peak-MAPE", "Peak-Timing", "FHV", "NSE"]]
merged = ev.merge(lib_t, on=["model_name", "set"], suffixes=("", "_lib"))
merged["abs_vol"] = merged["event_volume_error_mean_pct"].abs()
merged["abs_FHV"] = merged["FHV"].abs()
rank_cols = {"event_peak_abs_error_mean_pct": True, "event_timing_abs_error_mean_h": True, "event_event_NSE_median": False,
             "abs_vol": True, "Peak-MAPE": True, "abs_FHV": True}
merged["mean_rank"] = pd.DataFrame({c: merged[c].rank(ascending=asc) for c, asc in rank_cols.items()}).mean(axis=1)
peak_ranking = merged.sort_values("mean_rank")[["model_name", "set", "event_peak_abs_error_mean_pct", "event_peak_error_mean_pct",
                                                "event_timing_abs_error_mean_h", "event_event_NSE_median", "event_volume_error_mean_pct",
                                                "Peak-MAPE", "FHV", "NSE", "mean_rank"]].reset_index(drop=True)
peak_ranking.index += 1
display(peak_ranking.round(2))
peak_ranking.round(3).to_csv(common.RESULTS_DIR / "peaks_ranking_test.csv")'''))
    cells.append(new_code_cell('''lines = []
top = peak_ranking.iloc[0]
lines.append(f"- 첨두 종합 1위: **{top['model_name']} (Set {top['set']})** — 평균 |첨두 크기 오차| {top['event_peak_abs_error_mean_pct']:.1f}%, "
             f"평균 |시각 오차| {top['event_timing_abs_error_mean_h']:.1f} h, 사건 NSE 중앙값 {top['event_event_NSE_median']:.2f}.")
bias = merged.groupby("model_name", observed=True)["event_peak_error_mean_pct"].mean().sort_values()
lines.append(f"- 첨두를 가장 과소 추정하는 모델: {bias.index[0]} ({bias.iloc[0]:+.1f}%), 가장 과대/덜 과소: {bias.index[-1]} ({bias.iloc[-1]:+.1f}%). "
             f"전체 평균 {bias.mean():+.1f}% → {'대부분의 모델이 첨두를 과소 추정' if bias.mean() < -5 else '첨두 편향은 전반적으로 작음'}.")
timing = merged.groupby("model_name", observed=True)["event_timing_error_mean_h"].mean().sort_values()
lines.append(f"- 첨두 시각: 평균 시각 오차가 가장 작은(이른) 모델 {timing.index[0]} ({timing.iloc[0]:+.1f} h), 가장 늦은 모델 {timing.index[-1]} ({timing.iloc[-1]:+.1f} h).")
w20 = merged.groupby("model_name", observed=True)["event_peak_within_20pct_frac"].max().sort_values(ascending=False)
lines.append(f"- 15개 사건 중 첨두 크기 ±20% 이내 비율(최적 세트): 최고 {w20.index[0]} {w20.iloc[0]*100:.0f}%, 최저 {w20.index[-1]} {w20.iloc[-1]*100:.0f}%.")
hard = event_matrix("peak_error_pct").abs().mean(axis=1).sort_values(ascending=False)
lines.append(f"- 모든 모델이 가장 어려워한 사건: {hard.index[0]} (평균 |오차| {hard.iloc[0]:.0f}%), 가장 쉬운 사건: {hard.index[-1]} ({hard.iloc[-1]:.0f}%).")
display(Markdown("\\n".join(lines)))'''))

    cells.append(new_markdown_cell(f"""## 8. 해석: 모델별 홍수 재현 특성

{COMPARISON_NOTES.get("peaks", "_(학습이 끝난 뒤 작성)_").strip()}
"""))
    cells.append(new_markdown_cell("""## 9. Future work

- 첨두 특화 모델(PeakFocus, Exformer)의 **첨두 라벨 임계·손실 가중치·패치 크기** 튜닝 — 본 비교는 기본값만 사용.
- 첨두 과소 편향을 줄이기 위한 **손실 함수 변경**(NSE 손실, 첨두 가중 MSE, 분위 손실) 의 모든 모델 공통 적용.
- 2020년 대홍수(검증 기간, 4,395 m³/s)를 학습에 포함하는 **기간 분할 변경** 또는 교차검증으로 외삽 능력 재평가.
- 시드 앙상블로 사건별 오차의 **불확실성 구간** 제시; 확률적 head(CMAL/GMM) 로 첨두 예측 구간 산정.
- 선행시간(lead time) 1–6 h 예측(`forecast` 모드) 으로 확장해 실제 운영 관점의 첨두 예보 성능 비교.
""" ))
    return new_notebook(cells=cells, metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})


def write_all() -> list:
    written = []
    for model in MODEL_ORDER:
        path = HERE / nb_filename(model)
        nbformat.write(model_notebook(model), path)
        written.append(path)
    for path, nb in [(HERE / NB_OVERALL, comparison_overall_notebook()), (HERE / NB_PEAKS, comparison_peaks_notebook())]:
        nbformat.write(nb, path)
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_all():
        print("wrote", p.name)
