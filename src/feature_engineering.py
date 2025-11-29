def engineer_features(df):
    df['Symptom_Count'] = df['Health_Impacts'].apply(lambda x: len(str(x).split(',')))

    median = df['Avg_Daily_Screen_Time_hr'].median()
    q3 = df['Avg_Daily_Screen_Time_hr'].quantile(0.75)

    df['Is_Median_or_Higher'] = (df['Avg_Daily_Screen_Time_hr'] >= median).astype(int)
    df['Is_Q3_or_Higher'] = (df['Avg_Daily_Screen_Time_hr'] >= q3).astype(int)

    df['Is_Phone_User'] = (df['Primary_Device'] == "Smartphone").astype(int)
    df['Is_Tablet_User'] = (df['Primary_Device'] == "Tablet").astype(int)

    df['Age'] = df['Age'].replace(0, 1)
    df['ScreenTime_to_Median'] = df['Avg_Daily_Screen_Time_hr'] / median
    df['Age_Normalised_ScreenTime'] = df['Avg_Daily_Screen_Time_hr'] / df['Age']

    df['Risk_Score'] = (
        df['Avg_Daily_Screen_Time_hr']*0.5 +
        df['Is_Median_or_Higher']*1.8 +
        df['Is_Q3_or_Higher']*2.2 +
        df['Symptom_Count']*3 +
        df['Is_Phone_User']*1.5 +
        df['Is_Tablet_User']*1.7
    )

    return df
