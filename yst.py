import pandas as pd
import numpy as np
import statsmodels.api as sm

temperature_data = pd.read_csv('all_union_with_regions_upd.csv')
crime_data = pd.read_excel('data_Crime_final_upd.xlsx')

temperature_data['Datetime'] = pd.to_datetime(
    temperature_data['Datetime'],
    format='%d.%m.%Y %H:%M'
)

temperature_data['month'] = temperature_data['Datetime'].dt.month
winter_data = temperature_data[temperature_data['month'].isin([12, 1, 2])]

winter_avg_temp = (
    winter_data
    .groupby('region', as_index=False)['T']
    .mean()
)

crime_avg_rate = (
    crime_data
    .groupby('region', as_index=False)['crime_rate']
    .mean()
)

merged_data_winter = pd.merge(winter_avg_temp, crime_avg_rate, on='region')

def remove_outliers(df, column_x, column_y):
    Q1_x = df[column_x].quantile(0.25)
    Q3_x = df[column_x].quantile(0.75)
    IQR_x = Q3_x - Q1_x

    Q1_y = df[column_y].quantile(0.25)
    Q3_y = df[column_y].quantile(0.75)
    IQR_y = Q3_y - Q1_y

    filtered_df = df[
        (df[column_x] >= (Q1_x - 1.5 * IQR_x)) & (df[column_x] <= (Q3_x + 1.5 * IQR_x)) &
        (df[column_y] >= (Q1_y - 1.5 * IQR_y)) & (df[column_y] <= (Q3_y + 1.5 * IQR_y))
    ]
    return filtered_df

filtered_winter_crime = remove_outliers(
    merged_data_winter,
    'T',
    'crime_rate'
).reset_index(drop=True)

X_full = sm.add_constant(filtered_winter_crime['T'])
y_full = filtered_winter_crime['crime_rate']

model_full = sm.OLS(y_full, X_full).fit()
beta_full = model_full.params['T']

print("=== Исходная модель OLS ===")
print(model_full.summary())
print("\nИсходный коэффициент β1 (температура T):", beta_full)


n = len(filtered_winter_crime)

betas = []
regions_left_out = []

for i in range(n):
    df_j = filtered_winter_crime.drop(index=i)

    X_j = sm.add_constant(df_j['T'])
    y_j = df_j['crime_rate']

    res_j = sm.OLS(y_j, X_j).fit()
    betas.append(res_j.params['T'])
    regions_left_out.append(filtered_winter_crime.loc[i, 'region'])

betas = np.array(betas)

beta_mean = betas.mean()
jackknife_se = np.sqrt((n - 1) / n * np.sum((betas - beta_mean) ** 2))

beta_min = betas.min()
beta_max = betas.max()
max_dev = np.max(np.abs(betas - beta_full))

print("\n=== Результаты Jackknife по β1 (температура T) ===")
print(f"Число регионов: {n}")
print(f"Исходный β1 (полная выборка): {beta_full:.4f}")
print(f"Jackknife-среднее β1:        {beta_mean:.4f}")
print(f"Jackknife SE (std):          {jackknife_se:.4f}")
print(f"Мин. β1 при удалении 1 региона: {beta_min:.4f}")
print(f"Макс. β1 при удалении 1 региона: {beta_max:.4f}")
print(f"Максимальное отклонение от исходного β1: {max_dev:.4f}")

jackknife_table = pd.DataFrame({
    'region_left_out': regions_left_out,
    'beta_without_region': betas,
    'delta_from_full': betas - beta_full
}).sort_values('beta_without_region')

print("\n=== Влияние отдельных регионов на оценку β1 ===")
print(jackknife_table.to_string(index=False))
