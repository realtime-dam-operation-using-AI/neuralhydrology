neuralhydrology python library에서 제공하는 모델을 활용해서 댐의 hourly 유입량 예측에 대한 다양한 deep learning 방법을 비교하는 논문 작성을 위해서 아래와 같이 분석을 실행하고자 합니다.

1. 입력데이터는 
 - A. 유역평균 강수량 (202610151100_YDDam_average_rf_daily_input_A.csv, 202610151100_YDDam_average_rf_hourly_input_A.csv)
 - B. 관측소별 강수량 (202610151100_YDDam_rfs_daily_input_B.csv, 202610151100_YDDam_rfs_hourly_input_B.csv) 
 - C. 위의 A+B 3개의 입력 데이터 선정
 -> 입력자료는 기본적으로는 hourly 유입량 예측분석만 실행하며, MTS-LSTM의 경우만 daily 데이터를 활용
2. 예측을 위한 Target 데이터는 202610151100_YDDam_inflow_hourly_target.csv 입니다.

3. 1, 2의 데이터 세팅에 대해 CUDA-LSTM (LSTM으로 표기), GRU, Transformer, MTS-LSTM, MC-LSTM, ODE-LSTM, mamba, x_lstm, peak_transformer, peakfocus, exformer 총 11개 대해 각 모델별 분석 특성과 예측성능, 분석시간 등을 연계하여 설명하는 논문을 작성하고자 합니다.

4. 하이퍼 파라미터 튜닝은 Epochs을 최대 50번, 최소 30번으로 하고 early stopping 적용하기

5. 각 모델별 training, validation, testing 세트에 대한 비교 그래프를 과학논문에 적합하게 sticks, legends, 글자, 숫자 등이 잘 확인되게 크게 plotting을 만들어주세요.

6. future work로 하이퍼파라미터 튜닝에 대한 변경을 통해 모델별 비교를 하는 것을 작성하고자 합니다.

7. 위의 내용은 examples/07-CSV-Dam-Inflow 폴더에 데이터를 활용해 각 모델별 Jupyter Notebooks과 2개의 비교 Jupyter Notebooks(전체기간 성능비교, peaks에 대한 성능비교)를 작성하고, 각 모델별 Jupyter notebooks에는 논문의 methodology에 모델에 설명할 수 있도록 각 모델별로 일관되게 이해하기 쉽도록 모델에 대한 설명과 분석결과에 대한 설명을 작성해주세요. 그리고 비교 Jupyter Noteboook에는 각 모델별 hourly 강우-유출을 분석하는 홍수량 재현 특성과 성능결과를 연계하여 설명해주세요. 특히 peaks 성능비교 jupyter notebook은 performance metrics에 대한 상세한 설명과 함께 이해하기 쉽도록 plots을 생성해주세요.