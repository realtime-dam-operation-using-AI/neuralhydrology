"""Model descriptions (Korean, markdown) used in the per-model notebooks and the comparison notebooks.

Each entry of :data:`MODEL_DOCS` has the same four parts so that the methodology section of the paper can be written
consistently for all ten models:

* ``summary``      one-sentence characterisation (used in tables / comparison notebooks)
* ``description``  architecture and working principle (what the model computes, which inductive bias it has)
* ``hydrology``    what this means for hourly rainfall-runoff / dam-inflow modelling
* ``settings``     how the model is configured in this study and why
* ``references``   key references
"""

MODEL_DOCS = {
    # ------------------------------------------------------------------------------------------------------------
    "lstm": {
        "summary": "표준 LSTM(cuDNN 구현). 강우를 시간 순서대로 읽어 셀 상태에 유역 저류를 기억하는 기준(baseline) 모델.",
        "description": r"""
**LSTM (Long Short-Term Memory; NeuralHydrology `cudalstm`)**

LSTM은 입력 시계열을 한 시간 단계씩 순서대로 읽으면서 **은닉 상태 $h_t$** 와 **셀 상태 $c_t$** 를 갱신하는 순환신경망(RNN)이다.
셀 상태는 세 개의 게이트(입력·망각·출력)로 제어되며, 각 시간 단계에서 다음과 같이 계산된다.

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \quad
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f), \quad
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde c_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c), \qquad
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad
h_t = o_t \odot \tanh(c_t)
\end{aligned}
$$

마지막 시간 단계의 $h_t$ 가 선형 출력층(head)을 거쳐 그 시각의 유입량 예측값이 된다. `cudalstm`은 PyTorch의 cuDNN LSTM 커널을 그대로
사용하므로 같은 구조의 모델 중 학습 속도가 가장 빠르다. `initial_forget_bias`는 학습 초기에 망각 게이트를 열어(값 3) 긴 시간 의존성을 쉽게
학습하도록 돕는다.
""",
        "hydrology": r"""
- 셀 상태 $c_t$는 **유역 저류량(토양수분·지하수·하도저류)** 에 해당하는 역할을 학습한다. 망각 게이트는 감수(recession), 입력 게이트는 강우의
  침투·저류, 출력 게이트는 유출 발생에 대응한다는 해석이 가능하다(Kratzert et al., 2018).
- 입력이 강우뿐이므로 모델은 **선행 강우의 누적과 소멸**만으로 기저유출과 홍수 반응을 동시에 표현해야 한다. 720시간(30일) 창은 한 달 전의 강우까지
  기억에 반영할 수 있는 길이다.
- 모든 비교의 기준점이다. 다른 모델의 성능은 이 LSTM 대비 얼마나 개선/악화되는지로 해석한다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `hidden_size` | 64 | 단일 유역·소수 입력에 충분한 크기. 모든 순환형 모델에 동일 적용 |
| `initial_forget_bias` | 3 | 학습 초기 장기 기억 유지 |
| `output_dropout` | 0.4 | 출력층 앞 드롭아웃(과적합 억제) |
| `seq_length` / `predict_last_n` | 720 / 1 | 30일 선행 강우로 현재 시각 유입량 1개 예측 |
""",
        "references": """
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using LSTM networks. *HESS*, 22, 6005–6022.
- Kratzert, F. et al. (2022). NeuralHydrology — A Python library for deep learning research in hydrology. *JOSS*, 7(71), 4050.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "gru": {
        "summary": "LSTM보다 게이트가 하나 적은 경량 순환 모델. 셀 상태 없이 은닉 상태만으로 저류를 표현.",
        "description": r"""
**GRU (Gated Recurrent Unit; NeuralHydrology `gru`)**

GRU는 LSTM의 입력·망각 게이트를 하나의 **갱신 게이트 $z_t$** 로 합치고 셀 상태를 없앤 단순화된 순환 구조이다.

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}), \qquad r_t = \sigma(W_r x_t + U_r h_{t-1}) \\
\tilde h_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1})), \qquad
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}
$$

같은 `hidden_size`에서 파라미터 수가 LSTM의 약 3/4이고, 출력 게이트가 없어 은닉 상태가 곧 출력이다. 리셋 게이트 $r_t$는 과거 상태를 얼마나
무시하고 새 입력에 반응할지를 결정한다.
""",
        "hydrology": r"""
- 갱신 게이트 $z_t$가 "이전 저류를 유지할지, 새 강우로 갱신할지"를 한 번에 결정하므로 **빠른 홍수 반응**은 잘 표현하지만, LSTM처럼 출력과
  분리된 장기 저류(셀 상태)가 없어 **느린 감수·기저유출**의 표현력이 다소 제한될 수 있다.
- 파라미터가 적고 연산이 단순해 학습 시간이 짧다. 단일 유역·소량 데이터에서는 LSTM과 비슷하거나 더 안정적인 결과를 내는 경우가 많다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `hidden_size` | 64 | LSTM과 동일 |
| `output_dropout` | 0.4 | LSTM과 동일 |
| `seq_length` / `predict_last_n` | 720 / 1 | LSTM과 동일 |
""",
        "references": """
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*.
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv:1412.3555*.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "transformer": {
        "summary": "자기주의(self-attention) 인코더. 720시간 창의 모든 시각 쌍을 직접 비교해 중요한 강우 시점을 가중.",
        "description": r"""
**Transformer (encoder-only; NeuralHydrology `transformer`)**

순환 구조 대신 **자기주의(self-attention)** 로 시계열을 처리한다. 입력 $x_t$(강우)는 먼저 임베딩 네트워크(`dynamics_embedding`, 64차원)를
거치고, 위치 인코딩(sinusoidal, `sum`)이 더해진 뒤 $L$개의 인코더 층을 통과한다. 각 층은 다중 헤드 자기주의와 피드포워드 네트워크로 구성된다.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \qquad Q = XW_Q,\; K = XW_K,\; V = XW_V
$$

창의 마지막 시각에 대응하는 인코더 출력이 head를 거쳐 유입량이 된다. 순환 모델과 달리 720개 시각 전체가 한 번에 병렬 처리되며, 어떤 과거
시각이 현재 유입량에 기여하는지가 주의 가중치로 드러난다.
""",
        "hydrology": r"""
- 주의 메커니즘은 **창 안의 특정 강우 사상(예: 3일 전의 집중호우)** 을 거리와 무관하게 직접 참조할 수 있어 지체시간이 긴 유역에 유리하다.
- 반면 **누적·감수 같은 상태 전이**를 명시적으로 모델링하지 않고 위치 인코딩에 의존하므로, 데이터가 적을 때 저유량 구간이 불안정하거나
  LSTM보다 과적합되기 쉽다는 보고가 많다.
- 연산량이 시퀀스 길이의 제곱($720^2$)에 비례하므로 epoch당 학습 시간이 순환 모델보다 길다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `dynamics_embedding` | fc, 64, tanh | 입력 차원을 헤드 수(4)로 나누어 떨어지는 64차원으로 올림 |
| `transformer_nheads` / `nlayers` | 4 / 2 | 작은 데이터에 맞춘 얕은 인코더 |
| `transformer_dim_feedforward` | 128 | 은닉 64의 2배 |
| `transformer_dropout` / `positional_dropout` | 0.1 / 0.1 | 표준값 |
| `transformer_positional_encoding_type` | sum | 임베딩에 위치 인코딩을 더함 |
""",
        "references": """
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
- Kratzert, F. et al. (2022). NeuralHydrology (transformer 구현). *JOSS*.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "mtslstm": {
        "summary": "일 단위 LSTM이 365일을 읽은 뒤 상태를 시간 단위 LSTM 가지에 넘겨 최근 14일을 시간 해상도로 처리하는 다중 시간규모 LSTM.",
        "description": r"""
**MTS-LSTM (Multi-Timescale LSTM; Gauch et al., 2021; NeuralHydrology `mtslstm`)**

하나의 모델이 **여러 시간 해상도의 입력을 함께 사용**하도록 설계된 LSTM이다. 본 연구에서는 두 개의 가지(branch)를 쓴다.

1. **일 단위 가지 (1D)**: 지난 365일의 **일 강우**(`*_daily`)를 LSTM$_{\mathrm{1D}}$가 읽어 장기 저류 상태를 만든다.
2. **상태 전달**: 시간 단위 창이 시작되는 시점의 $(h, c)$ 를 선형 변환(`transfer_mtslstm_states: linear`)해 시간 단위 가지의 초기 상태로 넘긴다.
3. **시간 단위 가지 (1h)**: 최근 336시간(14일)의 **시강우**(`*_hourly`)를 LSTM$_{\mathrm{1h}}$가 읽어 마지막 24시간의 유입량을 예측한다.

각 가지는 자기 해상도의 head를 가지며(`shared_mtslstm: false`), `tie_frequencies` 정규화 항은 "시간 예측 24개의 평균 ≈ 일 예측"이 되도록
두 가지를 묶는다. 모델이 일 단위와 시간 단위 예측을 동시에 내지만, 비교에서는 **시간 단위 예측만** 평가한다.
""",
        "hydrology": r"""
- 본 연구에서 **유일하게 일 단위 자료를 함께 쓰는 모델**이다. 장기(1년) 강우 이력은 일 해상도로, 홍수 반응은 시간 해상도로 다루므로
  시간 모델이 720시간 창으로는 볼 수 없는 **계절적 선행 습윤 상태**까지 반영할 수 있다.
- 시간 단위 가지가 하루에 24시간을 한 번에 예측하므로 자료 샘플 수가 1/24로 줄어 epoch이 빠르다. 대신 날짜 경계에서 예측이 이어 붙여지는
  구조라 자정 부근의 연속성은 `tie_frequencies` 와 상태 전달이 담당한다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `use_frequencies` | [1D, 1h] | 일·시간 두 해상도 |
| `seq_length` | {1D: 365, 1h: 336} | 1년 일 강우 + 14일 시강우 |
| `predict_last_n` | {1D: 1, 1h: 24} | 하루에 시간 예측 24개 |
| `forcings` | [hourly, daily] | `hourlycsv`가 일 파일을 함께 읽고 변수명에 `_daily`/`_hourly` 접미사를 붙임 |
| `transfer_mtslstm_states` | h, c: linear | 가지 간 상태 전달 |
| `regularization` | tie_frequencies | 해상도 간 일관성 |
| `hidden_size` 등 | LSTM과 동일 | |
""",
        "references": """
- Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., & Hochreiter, S. (2021). Rainfall–runoff prediction at multiple timescales with a single LSTM. *HESS*, 25, 2045–2062.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "mclstm": {
        "summary": "질량보존 LSTM. 강우(mm/h)를 질량으로 받아 셀에 저장·재분배·방출하므로 입력 강우량을 초과하는 유출을 낼 수 없음.",
        "description": r"""
**MC-LSTM (Mass-Conserving LSTM; Hoedt et al., 2021; NeuralHydrology `mclstm`)**

LSTM의 셀 상태를 **보존량(질량)** 으로 재해석한 모델이다. 하나의 **질량 입력** $x_t^{m}$(강우, mm/h)과 보조 입력 $x_t^{a}$(표준화된 다른
강우 변수)를 받아 다음을 계산한다.

$$
\begin{aligned}
\mathbf{i}_t &= \mathrm{softmax}(\cdot) \;\; \text{(입력 게이트: 질량을 셀들에 분배, 합 = 1)} \\
\mathbf{R}_t &= \text{행 정규화된 재분배 행렬 (셀 간 질량 이동)} \\
\mathbf{o}_t &= \sigma(\cdot) \;\; \text{(출력 게이트: 각 셀에서 내보낼 비율)} \\
\mathbf{c}_t &= (1 - \mathbf{o}_t) \odot \left(\mathbf{R}_t^\top \mathbf{c}_{t-1} + \mathbf{i}_t x_t^{m}\right), \qquad
\hat y_t = \sum_{k \ne \text{trash}} \big[\mathbf{o}_t \odot (\mathbf{R}_t^\top \mathbf{c}_{t-1} + \mathbf{i}_t x_t^{m})\big]_k
\end{aligned}
$$

모든 연산이 질량을 보존하므로 **(입력 누적) = (저류) + (출력 누적) + (trash cell)** 이 항상 성립한다. trash cell은 증발산처럼 유출로
나타나지 않는 손실을 흡수하는 특수 셀이다. 출력은 head 없이 질량 유출의 합으로 정의된다.
""",
        "hydrology": r"""
- **물수지를 구조적으로 보장**한다. 학습 자료에 없는 극한 강우에서도 강우량보다 큰 유출을 만들 수 없어 외삽이 물리적으로 안전하다.
- 질량 입력과 목표가 같은 단위여야 하므로 유입량을 **mm/h**로 변환(`inflow_mm = Q × 3.6 / 930.42 km²`)해 학습하고, 평가 시 다시 m³/s로
  되돌린다. 입력·목표 모두 **표준화하지 않는다**(`custom_normalization: none`).
- 강우가 하나의 스칼라여야 하므로 입력세트 B에서는 7개 관측소의 **산술평균**(`rf_stn_mean`)을 질량 입력으로, 관측소별 값은 보조(게이트)
  입력으로 쓴다. 세트 A/C에서는 유역평균 강우가 질량 입력이다.
- 손실이 표준화되지 않은 mm/h 단위로 계산되므로 다른 모델과 **학습 손실 값의 크기를 직접 비교할 수 없다**(학습곡선 해석 시 주의).
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `mass_inputs` | rf_avg (A, C) / rf_stn_mean (B) | 질량 입력은 하나여야 함 |
| `dynamic_inputs` | 나머지 강우 변수(보조 입력) | 게이트 계산에만 사용 |
| `target_variables` | inflow_mm | 질량 입력과 같은 단위(mm/h) |
| `custom_normalization` | 질량 입력·목표: none | 질량보존을 위해 표준화 금지 |
| `hidden_size` | 64 | 셀 64개(그중 1개는 trash cell) |
| `initial_forget_bias` | 3 | 학습 초기 출력 게이트를 열어 둠 |
""",
        "references": """
- Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., & Klambauer, G. (2021). MC-LSTM: Mass-conserving LSTM. *ICML*.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "odelstm": {
        "summary": "LSTM 은닉 상태를 상미분방정식(ODE)으로 연속시간 전개. 불규칙·다중 해상도 입력을 하나의 연속 상태로 처리.",
        "description": r"""
**ODE-LSTM (Lechner & Hasani, 2020; NeuralHydrology `odelstm`)**

LSTM 셀 뒤에 **ODE-RNN** 을 붙여, 두 관측 사이의 시간 간격 $\Delta t$ 동안 은닉 상태가 연속적으로 변하도록 만든 모델이다.

$$
h_{t} = \mathrm{ODESolve}\big(f_\theta,\; h_{t-1}^{\mathrm{LSTM}},\; \Delta t\big), \qquad
\frac{dh}{d\tau} = f_\theta(h, x_t)
$$

여기서 $f_\theta$는 작은 신경망이고, `ode_method: euler` 와 `ode_num_unfolds: 4` 는 각 시간 간격을 4단계의 오일러법으로 적분함을 뜻한다.
NeuralHydrology 구현은 MTS-LSTM과 같은 2해상도 입력(1D: 365일, 1h: 336시간)을 받되, 학습 중에는 오래된 구간을 **무작위 해상도(5D~1D)** 로
재집계해 모델이 다양한 $\Delta t$ 를 경험하게 한다. 마지막 24시간의 시간 단위 예측만 평가한다.
""",
        "hydrology": r"""
- 수문학적으로 **저류 상태의 연속시간 전개**(감수곡선 $dS/dt = -kS$ 와 유사)를 직접 학습하는 셈이어서, 관측 간격이 일정하지 않거나 해상도가
  섞인 자료에 적합하다.
- 본 자료는 시간 간격이 일정하므로 ODE의 장점이 크게 드러나지는 않지만, MTS-LSTM과 같은 입력(1년 일 강우 + 14일 시강우)을 **하나의
  연속 상태**로 처리한다는 점에서 MTS-LSTM의 대안으로 비교할 가치가 있다.
- 단, 이 모델은 **일 단위 자료를 쓰지 않는다**: 두 해상도 모두 `hourly` 파일의 시강우를 집계해 만든다(설정의 `forcings: [hourly]`).
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `use_frequencies` | [1D, 1h] | 두 해상도(시강우를 내부에서 일 단위로 집계) |
| `seq_length` / `predict_last_n` | {1D: 365, 1h: 336} / {1D: 1, 1h: 24} | MTS-LSTM과 동일 |
| `ode_method` / `ode_num_unfolds` | euler / 4 | 오일러법 4단계 적분 |
| `ode_random_freq_lower_bound` | 5D | 학습 시 무작위 재집계의 가장 거친 해상도 |
| `regularization` | tie_frequencies | 해상도 간 일관성 |
""",
        "references": """
- Lechner, M., & Hasani, R. (2020). Learning long-term dependencies in irregularly-sampled time series. *arXiv:2006.04418*.
- Gauch, M. et al. (2021). (ODE-LSTM과 MTS-LSTM의 수문 비교) *HESS*, 25, 2045–2062.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "mamba": {
        "summary": "선택적 상태공간모델(SSM). 입력에 따라 변하는 상태전이로 긴 시퀀스를 선형 시간에 처리.",
        "description": r"""
**Mamba (Selective State Space Model; Gu & Dao, 2023; NeuralHydrology `mamba`)**

Mamba는 연속시간 **상태공간모델(SSM)** 을 이산화한 시퀀스 층이다. 은닉 상태 $s_t \in \mathbb{R}^{N}$(`d_state`=16)이 다음과 같이 갱신된다.

$$
s_t = \bar A_t s_{t-1} + \bar B_t x_t, \qquad y_t = C_t s_t, \qquad
\bar A_t = \exp(\Delta_t A),\; \bar B_t = \Delta_t B_t
$$

핵심은 **선택성(selectivity)**: $\Delta_t, B_t, C_t$ 가 입력 $x_t$ 의 함수여서, 모델이 "이 입력은 기억하고 저 입력은 흘려보낼지"를 시점마다
결정한다. $\Delta_t$ 가 크면 상태가 새 입력으로 빠르게 덮어써지고, 작으면 이전 상태가 오래 유지된다. 순환 계산이지만 병렬 스캔으로 구현되어
시퀀스 길이에 **선형** 비용이며, 앞단의 짧은 1-D 합성곱(`d_conv`=4)이 국소 패턴을 잡는다. NeuralHydrology 구현은 입력을 `hidden_size`(64)로
선형 변환한 뒤 Mamba 블록(`expand`=2 → 내부 폭 128) 하나와 head로 구성된다.
""",
        "hydrology": r"""
- 입력 의존적 시간 상수 $\Delta_t$ 는 수문학적으로 **강우가 있을 때는 빠르게, 없을 때는 느리게 반응하는 저류** 와 잘 맞는다. 즉 게이트
  없이도 LSTM의 망각 게이트와 비슷한 역할을 한다.
- 긴 창(720시간)을 트랜스포머보다 훨씬 적은 메모리로 처리하므로 **시간 해상도 장기 입력**에 실용적이다.
- 상대적으로 새로운 구조라 수문 분야 적용 사례가 적다. 본 비교에서 LSTM 대비 어떤 유량 구간에서 차이가 나는지가 관심사이다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `hidden_size` (d_model) | 64 | 다른 모델과 동일 |
| `d_state` / `d_conv` / `expand` | 16 / 4 / 2 | mamba_ssm 기본값 |
| `output_dropout` | 0.4 | 동일 |
""",
        "references": """
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "xlstm": {
        "summary": "확장 LSTM. 지수 게이트와 행렬 메모리(mLSTM)·스칼라 메모리(sLSTM) 블록을 쌓아 LSTM의 저장 용량과 병렬성을 개선.",
        "description": r"""
**xLSTM (Extended LSTM; Beck et al., 2024; NeuralHydrology `x_lstm`)**

LSTM의 두 가지 한계(저장 용량, 게이트의 시그모이드 포화)를 고친 두 종류의 블록을 쌓은 모델이다.

- **mLSTM 블록**: 셀 상태를 벡터가 아닌 **행렬 $C_t \in \mathbb{R}^{d\times d}$** 로 두고, 키·값 쌍을 외적으로 저장한다.
  $C_t = f_t C_{t-1} + i_t\, v_t k_t^\top$, $h_t = o_t \odot \frac{C_t q_t}{\max(|n_t^\top q_t|, 1)}$.
  시간 단계 간 은닉 상태 의존이 없어 **완전 병렬** 학습이 가능하다(주의 메커니즘과 유사한 연상 기억).
- **sLSTM 블록**: 기존 LSTM에 **지수 게이트** $i_t = \exp(\cdot)$ 와 정규화 상태 $n_t$ 를 도입해 "저장 결정을 나중에 번복"할 수 있게 하고,
  다중 헤드로 메모리를 섞는다. 순환 의존이 남아 있어 CUDA 커널로 구현된다.

본 연구는 2개 블록(`xlstm_num_blocks: 2`, 두 번째가 sLSTM)을 4개 헤드로 쓴다. 입력은 선형 변환으로 64차원에 맞춘다.
""",
        "hydrology": r"""
- 행렬 메모리는 **여러 강우 사상을 서로 다른 키로 동시에 저장**할 수 있어, 선행 사상이 겹치는 장마철 복합 홍수에서 유리할 수 있다.
- 지수 게이트는 큰 입력(집중호우)에 대해 게이트가 포화되지 않아 **첨두 반응의 크기**를 더 잘 표현할 가능성이 있다.
- 720시간 창에 대한 mLSTM의 병렬 커널은 **GPU 메모리를 크게**(≈ 21 GB) 사용하므로 학습 시 다른 run 과 GPU를 공유하지 않는다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `hidden_size` | 64 | 동일 |
| `xlstm_num_blocks` / `xlstm_slstm_at` | 2 / [1] | mLSTM → sLSTM 순서 |
| `xlstm_heads` | 4 | 다중 헤드 메모리 |
| `xlstm_kernel_size` / `xlstm_proj_factor` | 4 / 1.3 | 공식 구현 기본값 |
""",
        "references": """
- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended long short-term memory. *NeurIPS*.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "peakfocus": {
        "summary": "첨두 위치 분류(MSM-PL)와 유량 회귀를 함께 학습하고, 위치 인식 디코더(LAD)가 첨두 특징으로 교차주의를 게이팅.",
        "description": r"""
**PeakFocus (joint peak localization + intensity regression; NeuralHydrology `peakfocus`)**

"첨두를 어디서, 얼마나 크게"를 **두 개의 과제로 분리해 동시에 학습**하는 모델이다.

- **인코더**: 잔차 MLP 블록(`mlp_layers`=2)으로 각 시각의 특징을 만든다.
- **MSM-PL (Multi-Scale Mixing Peak Locator)**: 특징을 2단계 피라미드(`n_scales`=2)로 다운/업샘플해 섞은 뒤 시각별 **첨두 로짓**을 낸다.
  목표 창(마지막 24시간)에서 "±3시간 안의 최대이면서 표준화 값 > 1"인 시각을 첨두로 라벨링하고, 가우시안으로 ±1시간을 부드럽게 한 soft label에
  **focal loss**($\alpha$=0.25, $\gamma$=2)를 적용한다 → `peak_focal` 정규화 항.
- **LAD (Location-Aware Decoder)**: 디코더의 교차주의가 첨두 특징으로 **게이팅**되어, 첨두로 판정된 시각의 정보가 유량 회귀에 더 크게
  반영된다.
- **손실**: $\mathcal{L} = \mathrm{MSE}_{\text{전체}} + 1.0\cdot\mathrm{Focal}_{\text{첨두}} + 1.0\cdot\mathrm{MSE}_{\text{첨두 시각만}}$
  (`peak_weighted_mse`는 첨두 라벨 시각에서만 평균한 MSE로, 첨두 크기가 다수의 비첨두 시각에 희석되지 않게 한다).

`predict_last_n: 24` 로 마지막 24시간을 모두 예측해 첨두 head를 학습하지만, 평가는 다른 모델과 같이 **각 창의 마지막 시각 예측**만 사용한다.
원 구현은 입력창을 창별 평균·표준편차로 정규화한 뒤 목표 변수의 과거 통계로 출력을 역정규화하는데, 본 설정에서는 목표(유입량)가 입력이 아니어서
역정규화가 불가능하므로 창별 정규화를 끄고(`peakfocus_instance_norm: false`) NeuralHydrology 의 전역 표준화 입력을 그대로 사용한다.
""",
        "hydrology": r"""
- 일반 MSE 학습은 전체 시간의 85%를 차지하는 저유량에 지배되어 첨두가 과소되기 쉽다. PeakFocus는 **첨두 시각을 별도 과제로 학습**해 이
  불균형을 직접 다룬다.
- 분류 head는 "첨두가 임박했는가"를 출력하므로, 댐 운영 관점의 **홍수 경보**로도 쓸 수 있는 부산물이다.
- 라벨 임계(표준화 1.0 ≈ 평균 + 1σ ≈ 114 m³/s)와 창(±3 h)은 시간 자료에 맞춰 정한 값이며, 라벨 정의가 성능에 민감하다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `peakfocus_d_model` / `n_heads` / `d_ff` | 64 / 4 / 128 | Transformer와 동일 |
| `peakfocus_d_layers` / `mlp_layers` / `n_scales` | 1 / 2 / 2 | 논문 설정 |
| `peakfocus_if_msm_pl` / `if_lad` | true / true | 두 구성요소 모두 사용 |
| `peakfocus_instance_norm` | false | 원 논문의 입력창 정규화는 목표 이력으로 역정규화하는 전제. 여기서는 목표가 입력이 아니므로 정규화를 끄고 전역 표준화 입력을 그대로 사용 (켜면 강우 절대 크기 정보가 사라져 학습 불가) |
| `predict_last_n` | 24 | 첨두 head 학습용 목표 창 |
| `regularization` | peak_focal 1.0, peak_weighted_mse 1.0 | 3중 손실 |
| `peak_label_window` / `threshold` | 3 / 1.0 | ±3 h 최대, 표준화 > 1 |
| `peak_soft_label_sigma` / `tolerance` | 2.0 / 1 | soft label |
| `peak_focal_alpha` / `gamma` | 0.25 / 2.0 | focal loss |
""",
        "references": """
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
- PeakFocus 참조 구현(`proposed_model.py`: Multi_Scale_Mixing_Peak_Locator, Location_Aware_Decoder_Layer)의 NeuralHydrology 이식.
""",
    },
    # ------------------------------------------------------------------------------------------------------------
    "exformer": {
        "summary": "극한 적응 트랜스포머. 패치 단위 희소 주의(국소+주기+극한 마스크)와 계절/추세 분해로 극한 강우 패치를 서로 연결.",
        "description": r"""
**Exformer (Extreme-Adaptive Transformer; NeuralHydrology `exformer`)**

극한값(extreme) 예측을 위해 설계된 **패치 기반·채널 독립** 트랜스포머이다.

1. **RevIN + 계절/추세 분해**: 입력을 가역 정규화한 뒤 이동평균(커널 13, 25)으로 추세와 계절(잔차) 성분으로 나눈다. 추세는 선형층으로,
   계절 성분은 인코더로 처리해 더한다.
2. **패치화**: 720시간 창을 24시간(1일) 패치 30개로 나누고, 각 패치를 $4 \times 24 = 96$ 차원 토큰으로 임베딩한다. 주의는 시각이 아닌
   **패치 사이**에서 계산된다.
3. **극한 적응 희소 주의(Dozer 마스크 `dozer_ext_0`)**: 패치 $i$가 참조할 수 있는 패치를 세 가지로 제한한다.
   - *Local*: $|i-j| \le 2$ (최근 이틀 전후),
   - *Stride*: $j \equiv i \pmod 7$ (주 단위 주기),
   - *Extreme*: 극한 라벨이 같은 패치. 극한 라벨은 지정한 강우 변수(`exformer_extreme_variable`)가 표준화 값 2.0을 넘는 시각이 패치 안에
     1개 이상(`patch_thres`) 있으면 1이 된다.
   따라서 **과거 어느 시점의 극한 강우 패치든 현재 극한 패치와 직접 연결**된다.
4. 마지막 패치의 출력이 head를 거쳐 유입량이 된다.
""",
        "hydrology": r"""
- 극한 마스크는 "지금 집중호우가 오고 있으면 과거 집중호우 때의 유역 반응을 참조하라"는 **사상 기반 유추**를 구조에 넣은 것이다.
- 패치 단위 처리로 주의 비용이 $30^2$ 으로 줄어 긴 창을 싸게 다루지만, **시간 단위의 미세한 지체**(수 시간)는 패치 내부 임베딩이 담당하므로
  첨두 시각 정밀도는 순환 모델보다 낮을 수 있다.
- 극한 라벨 변수는 하나의 스칼라 강우여야 하므로 입력세트 B에서는 관측소 평균(`rf_stn_mean`)을 추가 입력으로 넣어 라벨에 쓴다.
""",
        "settings": r"""
| 항목 | 값 | 이유 |
|---|---|---|
| `exformer_patch_size` / `embed_dim` | 24 / 4 | 1일 패치, 토큰 96차원 |
| `exformer_n_heads` / `encoder_depth` / `d_ff` | 4 / 2 / 32 | 얕은 인코더 |
| `exformer_local_window` / `stride` | 2 / 7 | ±2일 국소, 7일 주기 |
| `exformer_mask` / `patch_thres` | dozer_ext_0 / 1 | 극한 적응 마스크 |
| `exformer_extreme_variable` / `threshold` | rf_avg (A, C) / rf_stn_mean (B); 2.0 | 극한 라벨 정의 |
| `exformer_moving_avg` | [13, 25] | 분해 커널 |
""",
        "references": """
- Zhang, Z. et al. (2024). Dozerformer: Sequence adaptive sparse transformer for multivariate time series forecasting. *arXiv:2312.06874*.
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers (PatchTST). *ICLR*.
- Exformer 참조 구현(`build_model_util.py`, `flash_dozer.py`)의 NeuralHydrology 이식.
""",
    },
}

MODEL_FAMILY = {
    "lstm": "순환(게이트)", "gru": "순환(게이트)", "mtslstm": "순환(다중 시간규모)", "mclstm": "순환(질량보존)",
    "odelstm": "순환(연속시간)", "xlstm": "순환(확장 LSTM)", "mamba": "상태공간(SSM)",
    "transformer": "주의(트랜스포머)", "peakfocus": "주의(첨두 과제 결합)",
    "exformer": "주의(희소·패치 트랜스포머)",
}


def methodology_markdown(model: str) -> str:
    """Full methodology section (description + hydrology + settings + references) of one model."""
    d = MODEL_DOCS[model]
    return (f"## 1. 모델 설명 (Methodology)\n\n{d['description'].strip()}\n\n"
            f"### 1.1 수문학적 관점에서의 특징\n\n{d['hydrology'].strip()}\n\n"
            f"### 1.2 본 연구에서의 설정\n\n{d['settings'].strip()}\n\n"
            f"### 참고문헌\n\n{d['references'].strip()}\n")
