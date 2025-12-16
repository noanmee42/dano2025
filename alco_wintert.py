import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

w = pd.read_csv('all_union_with_regions_upd.csv')
alco = pd.read_excel('potreblenie_alco_2018_2020_filtered_28regions.xlsx')

w['Datetime'] = pd.to_datetime(w['Datetime'], format='%d.%m.%Y %H:%M')
w['month'] = w['Datetime'].dt.month
w['year'] = w['Datetime'].dt.year

winter = w[(w['month'].isin([12, 1, 2])) & (w['year'].between(2018, 2020))]

winter_avg = (
    winter.groupby('region')['T']
    .mean()
    .reset_index()
    .rename(columns={'T': 'winter_temp'})
)

alco['alco_mean'] = alco[['Потребление_2018',
                          'Потребление_2019',
                          'Потребление_2020']].mean(axis=1)
alco_mean = alco[['region', 'alco_mean']]

data = pd.merge(winter_avg, alco_mean, on='region', how='inner')

def remove_outliers_iqr(df, col_x, col_y):
    Q1_x = df[col_x].quantile(0.25)
    Q3_x = df[col_x].quantile(0.75)
    IQR_x = Q3_x - Q1_x

    Q1_y = df[col_y].quantile(0.25)
    Q3_y = df[col_y].quantile(0.75)
    IQR_y = Q3_y - Q1_y

    lower_x = Q1_x - 1.5 * IQR_x
    upper_x = Q3_x + 1.5 * IQR_x
    lower_y = Q1_y - 1.5 * IQR_y
    upper_y = Q3_y + 1.5 * IQR_y

    mask = (
        (df[col_x] >= lower_x) & (df[col_x] <= upper_x) &
        (df[col_y] >= lower_y) & (df[col_y] <= upper_y)
    )

    cleaned = df[mask]
    outliers = df[~mask]

    bounds = {
        'lower_x': lower_x,
        'upper_x': upper_x,
        'lower_y': lower_y,
        'upper_y': upper_y,
    }
    return cleaned, outliers, bounds

cleaned, outliers, bounds = remove_outliers_iqr(data, 'winter_temp', 'alco_mean')

print("Границы без выбросов:")
print(f"  winter_temp: [{bounds['lower_x']:.2f}, {bounds['upper_x']:.2f}]")
print(f"  alco_mean:   [{bounds['lower_y']:.2f}, {bounds['upper_y']:.2f}]")
print("\nВыбросы:")
print(outliers[['region', 'winter_temp', 'alco_mean']])

r = np.corrcoef(cleaned['winter_temp'], cleaned['alco_mean'])[0, 1]
print(f"Коэффициент корреляции Пирсона r = {r:.3f}")

plt.figure(figsize=(10, 6))
plt.scatter(cleaned['winter_temp'], cleaned['alco_mean'],
            label='Данные (без выбросов)')

coeffs = np.polyfit(cleaned['winter_temp'], cleaned['alco_mean'], 1)
poly = np.poly1d(coeffs)
x_vals = np.linspace(cleaned['winter_temp'].min(),
                     cleaned['winter_temp'].max(), 100)
y_vals = poly(x_vals)
plt.plot(x_vals, y_vals, label='Линия тренда')

plt.title('Зависимость потребления алкоголя от средней зимней температуры (2018–2020, без выбросов)')
plt.xlabel('Средняя зимняя температура (°C)')
plt.ylabel('Среднее потребление алкоголя (л на душу, 2018–2020)')
plt.grid(True)
plt.legend()
plt.show()
