import pandas as pd

def load_dataset():
    path = "data/raw/Indian_Kids_Screen_Time.csv"
    df = pd.read_csv(path)
    return df

def create_labels(df):
    df['Eye_Strain_Present'] = df['Health_Impacts'].apply(
        lambda x: 1 if 'Eye Strain' in str(x) else 0
    )

    df['ScreenTime_Class'] = pd.cut(
        df['Avg_Daily_Screen_Time_hr'],
        bins=[0, 1.5, 3.5, df['Avg_Daily_Screen_Time_hr'].max() + 1],
        labels=[0,1,2]
    ).astype(int)

    df['Eye_Strain_Risk_Level'] = df['ScreenTime_Class']
    df.loc[df['Eye_Strain_Present'] == 1, 'Eye_Strain_Risk_Level'] = 2

    return df
