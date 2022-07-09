import pandas as pd

# Считывание таблицы
NHL = pd.read_csv("Data/NHL.csv", delimiter=',', encoding='latin-1')
# Удаление лишних столбцов
# NHL = NHL.drop(['Rk', 'PS', 'EV', 'PP', 'SH', 'GW', 'EV.1', 'PP.1', 'SH.1', 'TOI', 'HART', 'Votes'], axis=1)
NHL = NHL.drop(['Rk', 'Age', 'Tm', 'PS', 'EV', 'PP', 'SH', 'GW', 'EV.1', 'PP.1', 'SH.1', 'TOI', 'HART', 'Votes'],
               axis=1)
# Форматирование столбца "Player"
NHL['Player'] = NHL['Player'].str.extract(r'([^\\]+(?=\\))', expand=False)

gp = 20  # Количество проведенных игр
season = 2018  # Сезон

# Удаление строк, не удовлетворяющих условиям
NHL = NHL.drop(NHL[(NHL.GP < gp) | (NHL.Season == season) | (NHL.Season < season - 1)].index)
# NHL = NHL.drop(NHL[(NHL.GP < gp) | (NHL.Season < season)].index)
# Удаление стообца "Season"
NHL = NHL.drop('Season', axis=1)
# Обновление индексов
NHL = NHL.reset_index(drop=True)
# Сохранение хаактеристик
Stats = NHL.columns.tolist()
# Преобразование DataFrame в массив Numpy
NHL = NHL.to_numpy()
