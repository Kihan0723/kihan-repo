import numpy as np
import matplotlib.pyplot as plt
import matrixprofile as mp

# 시계열 데이터 생성
np.random.seed(0)  # 결과의 일관성을 위해 seed 설정
time_series = np.random.normal(loc=0, scale=1, size=1000)  # 정규 분포를 따르는 임의의 데이터
time_series[300:400] += 5  # 데이터에 이상치 추가

# Matrix Profile 계산
window_size = 50  # 패턴을 비교할 윈도우 크기
profile = mp.compute(time_series, windows=window_size)
profile = mp.discover.motifs(profile, k=1)  # 가장 빈번한 패턴 찾기

# 이상치 탐지 (가장 낮은 Matrix Profile 값을 가진 지점을 이상치로 간주)
threshold = 2  # 임계값 설정
anomalies = np.where(profile['mp'] > threshold)[0]

# 결과 시각화
plt.figure(figsize=(15, 5))
plt.plot(time_series, label='Time Series')
plt.scatter(anomalies, time_series[anomalies], color='r', label='Anomalies')
plt.legend()
plt.show()

# 새로운 기능 추가 필요
#