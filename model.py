import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import matplotlib.pyplot as plt

"""В качестве модели сразу решил использовать деревья, так как они чаще всего лучше 
всего показывают себя на табличных данных. В качестве возможных кандидатов рассматривал 
градиентный бустинг и случайный лес. Первый вариант моделей не сохранился. Я попробовал обучить их исключительно на 
информации о самой мутации ('Mutation(s)_cleaned'). Результат получился примерно одинаковый, что на бустинге, что на 
случайном лесе, более-менее неплохо делались предсказания на классе с положительным знаком реакции, но на отрицательном 
была достаточно сильная просадка. В классах наблюдается достаточно серьезный дисбаланс. Можно было попробовать 
сбалансировать классы, но я решил для начала добавить больше фичей. Не могу сказать, что я силен в биологии, 
но добавил те фичи, которые показались мне существенными на первый взгляд. Попробовал обучить и получил 
достаточно хороший результат на модели градиентного бустинга, который все еще дает небольшую просадку на отрицательном 
классе, но уже куда менее существенную. Модель случайного леса показала себя не очень хорошо, по сравнению с catboost."""

#Загрузка датасета
df = pd.read_csv('skempi_v2.csv', sep=';', encoding='utf-8')
#Удаление неиспользуемых в обучении столбцов, очистка от пропусков
df = df[['Mutation(s)_cleaned', 'Affinity_mut_parsed', 'Affinity_wt_parsed', 'Temperature', 'Method',
         'iMutation_Location(s)', 'Protein 1', 'Protein 2']].dropna()

#Константы для вычисления свободной энергии связывания
R = 1.987e-3
T = 298

#Вычисление свободных энергий связывания, их разности и знака разности (целевой переменной)
df['dG_mut'] = R * T * np.log(df['Affinity_mut_parsed'])
df['dG_wt'] = R * T * np.log(df['Affinity_wt_parsed'])
df['ddG'] = df['dG_mut'] - df['dG_wt']
df['ddG_sign'] = (df['ddG'] > 0).astype(int)

#Функция для парсинга мутации на изначальный белок, позицию и измененный белок
def parse_mutation(mutation):
    try:
        wild = mutation[:3]
        pos = mutation[3:-3]
        mut_res = mutation[-3:]
        return wild, pos, mut_res
    except:
        return None, None, None


#Разбитие мутации на три столбца при помощи parse_mutation()
df[['wt_res', 'pos', 'mut_res']] = df['Mutation(s)_cleaned'].apply(lambda x: pd.Series(parse_mutation(x)))

df = df.dropna(subset=['wt_res', 'pos', 'mut_res']) #Очистка от пропусков

#Список названий столбцов с фичами для обучения
features = [
    'Affinity_mut_parsed',
    'Affinity_wt_parsed',
    'Temperature',
    'wt_res',
    'pos',
    'mut_res',
    'Method',
    'Protein 1',
    'Protein 2',
    'iMutation_Location(s)'
]
target = 'ddG_sign' #Название столбца с целевой переменной

#Извлекает число из строки (для валидации подобных данных в таблице: "298(assumed)". В результате: 298.0)
def clean_temperature(temp):
    if isinstance(temp, str):
        match = re.search(r'\d+(?:\.\d+)?', temp)
        return float(match.group()) if match else None
    return temp  #если уже float

#Применение clean_temperature() к столбцу с температурой
df['Temperature'] = df['Temperature'].apply(clean_temperature)

#Разделение выборки на трейновую, валидационную и тестовую
X_train_all, X_test, y_train_all, y_test = train_test_split(df[features],
                                                            df[target], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

#Список столбцов с категориальными фичами для верной обработкой catboost
cat_features = ['wt_res', 'pos', 'mut_res', 'Method', 'Protein 1', 'Protein 2', 'iMutation_Location(s)']

#Создание и обучение градиентного бустинга для классификации
model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0)
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

#Построение графика обучения модели
results = model.get_evals_result()
plt.plot(results['learn']['Logloss'], label='Train')
plt.plot(results['validation']['Logloss'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
plt.legend()
plt.title('CatBoost learning curve')
plt.show()

#Проверка результатов на тестовой выборке
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
