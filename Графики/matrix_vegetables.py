import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

w_df     = pd.read_csv('all_union_with_regions_upd.csv')          
crime_df = pd.read_excel('data_Crime_final_upd.xlsx')             
alco_df  = pd.read_excel('potreblenie_alco_2018_2020_filtered_28regions.xlsx')   
food_df  = pd.read_excel('food_sheet1_upd.xlsx')                  

w_df['Datetime'] = pd.to_datetime(w_df['Datetime'], format='%d.%m.%Y %H:%M')
w_df['year'] = w_df['Datetime'].dt.year
w_df['month'] = w_df['Datetime'].dt.month

winter = w_df[(w_df['month'].isin([12, 1, 2])) & (w_df['year'].between(2018, 2020))]

winter_temp = (
    winter
    .groupby('region')['T']
    .mean()
    .reset_index()
    .rename(columns={'T': 'winter_avg_temp'})
)

alco_df['alcohol_consumption_avg'] = alco_df[
    ['Потребление_2018', 'Потребление_2019', 'Потребление_2020']
].mean(axis=1)

alco_region = alco_df[['region', 'alcohol_consumption_avg']]

base = pd.merge(winter_temp, alco_region, on='region', how='inner')

crime_cols = [
    'crime_rate',
    'crime_rate_severe',
    'crime_rate_econ',
]

crime_1820 = crime_df[crime_df['year'].between(2018, 2020)]
crime_region = (
    crime_1820
    .groupby('region')[crime_cols]
    .mean()
    .reset_index()
)

if 'Регион' in food_df.columns:
    food_df = food_df.rename(columns={'Регион': 'region'})

veg_cols = ['Овощи 2018', 'Овощи 2019', 'Овощи 2020']

food_region = (
    food_df
    .groupby('region')[veg_cols]
    .mean()
    .reset_index()
)

food_region['vegetables_consumption_avg'] = food_region[veg_cols].mean(axis=1)
food_region = food_region[['region', 'vegetables_consumption_avg']]

full = base.merge(crime_region, on='region', how='left')
full = full.merge(food_region, on='region', how='left')

full = full.dropna(subset=[
    'winter_avg_temp',
    'alcohol_consumption_avg',
    'vegetables_consumption_avg',
])

def remove_outliers_iqr(df, x_col, y_col):
    q1_x = df[x_col].quantile(0.25)
    q3_x = df[x_col].quantile(0.75)
    iqr_x = q3_x - q1_x
    lower_x = q1_x - 1.5 * iqr_x
    upper_x = q3_x + 1.5 * iqr_x

    q1_y = df[y_col].quantile(0.25)
    q3_y = df[y_col].quantile(0.75)
    iqr_y = q3_y - q1_y
    lower_y = q1_y - 1.5 * iqr_y
    upper_y = q3_y + 1.5 * iqr_y

    mask = (
        (df[x_col] >= lower_x) & (df[x_col] <= upper_x) &
        (df[y_col] >= lower_y) & (df[y_col] <= upper_y)
    )
    return df[mask].copy()

clean = remove_outliers_iqr(full, 'winter_avg_temp', 'alcohol_consumption_avg')

cols = [
    'crime_rate',
    'crime_rate_severe',
    'crime_rate_econ',
    'winter_avg_temp',
    'alcohol_consumption_avg',
    'vegetables_consumption_avg',
]

corr = clean[cols].corr()

name_map = {
    'crime_rate'                : 'Общая преступность',
    'crime_rate_severe'         : 'Тяжкие преступления',
    'crime_rate_econ'           : 'Экономические преступления',
    'winter_avg_temp'           : 'Средняя зимняя температура',
    'alcohol_consumption_avg'   : 'Потребление алкоголя',
    'vegetables_consumption_avg': 'Потребление овощей',
}

corr_ru = corr.rename(index=name_map, columns=name_map)

print(corr_ru)
print(
    'r(зима, алкоголь) =',
    corr_ru.loc['Средняя зимняя температура', 'Потребление алкоголя']
)

plt.figure(figsize=(8, 8))
sns.heatmap(corr_ru, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')

plt.title('Матрица корреляций: преступность, климат и потребление')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
