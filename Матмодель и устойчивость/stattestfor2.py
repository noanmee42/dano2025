import pandas as pd
import statsmodels.api as sm

temperature_data = pd.read_csv('all_union_with_regions_upd.csv')
crime_data = pd.read_excel('data_Crime_final_upd.xlsx')

temperature_data['Datetime'] = pd.to_datetime(temperature_data['Datetime'], format='%d.%m.%Y %H:%M')

temperature_data['month'] = temperature_data['Datetime'].dt.month

winter_data = temperature_data[temperature_data['month'].isin([12, 1, 2])]
winter_avg_temp = winter_data.groupby('region')['T'].mean().reset_index()

crime_avg_rate = crime_data.groupby('region')['crime_rate'].mean().reset_index()

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

filtered_winter_crime = remove_outliers(merged_data_winter, 'T', 'crime_rate')

X = filtered_winter_crime['T']
X = sm.add_constant(X)

y = filtered_winter_crime['crime_rate']

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
