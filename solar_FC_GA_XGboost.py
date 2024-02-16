import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
import random

# 데이터 불러오기
df = pd.read_csv('dataset.csv')

# 'Rad'와 'solar Generation'이 모두 0인 행 제거
data_filtered = df[(df['Rad'] != 0)]

# 피처와 타겟 분리
X = df[['Temperature', 'Humidity', 'WindSpeed', 'Cloud', 'Rad']]
y = df['solar Generation']

# 0-1 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# train-test 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 유전 알고리즘을 위한 하이퍼파라미터 설정 함수
def evalXGB(individual):
    learning_rate = individual[0]
    n_estimators = int(individual[1])
    max_depth = int(individual[2])
    min_child_weight = individual[3]
    gamma = individual[4]
    subsample = individual[5]
    colsample_bytree = individual[6]

    model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                             max_depth=max_depth, min_child_weight=min_child_weight,
                             gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return (mean_absolute_error(y_test, predictions),)

# 유전 알고리즘 설정
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_learning_rate", random.uniform, 0.01, 0.9)
toolbox.register("attr_n_estimators", random.randint, 50, 500)
toolbox.register("attr_max_depth", random.randint, 3, 20)
toolbox.register("attr_min_child_weight", random.uniform, 1, 10)
toolbox.register("attr_gamma", random.uniform, 0, 0.5)
toolbox.register("attr_subsample", random.uniform, 0.5, 1)
toolbox.register("attr_colsample_bytree", random.uniform, 0.5, 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_learning_rate, toolbox.attr_n_estimators, toolbox.attr_max_depth,
                  toolbox.attr_min_child_weight, toolbox.attr_gamma, toolbox.attr_subsample, 
                  toolbox.attr_colsample_bytree), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalXGB)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 유전 알고리즘 실행
population = toolbox.population(n=50)
ngen = 20
result = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.2, ngen=ngen, verbose=False)

# 최적의 하이퍼파라미터 선택
best_individual = tools.selBest(population, k=1)[0]

# 최적화된 하이퍼파라미터로 XGBoost 모델 학습
learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree = best_individual
model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                         min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, 
                         colsample_bytree=colsample_bytree)
model.fit(X_train, y_train)

def NMAE(true, pred, nominal):
    absolute_error = np.abs(true - pred)
    absolute_error /= nominal
    target_idx = np.where(true >= nominal*0.1)
    return 100*absolute_error.iloc[target_idx].mean()
    

# 예측 및 평가
predictions = model.predict(X_test)


# NMSE 및 R² 계산
nmse = NMAE(y_test, predictions, 56)
print(f'Normalized Mean Squared Error (NMSE): {nmse}')

r2 = r2_score(y_test, predictions)
print(f'R² score: {r2}')