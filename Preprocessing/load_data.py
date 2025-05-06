import pandas as pd

# Load cleaned data
df = pd.read_excel('CDD_structured_ml_ready.xlsx')  # or the new file name you saved

# Preview structure
print(df.shape)
print(df.dtypes)
df.head()

df['EVENT START DATE'] = pd.to_datetime(df['EVENT START DATE'])
df['EVENT END DATE'] = pd.to_datetime(df['EVENT END DATE'])
df['EVENT DURATION (DAYS)'] = (df['EVENT END DATE'] - df['EVENT START DATE']).dt.days


df['TOTAL HUMAN IMPACT'] = df[['FATALITIES', 'INJURED / INFECTED', 'EVACUATED']].sum(axis=1)


financial_cols = ['FEDERAL DFAA PAYMENTS', 'PROVINCIAL DFAA PAYMENTS', 
                  'PROVINCIAL DEPARTMENT PAYMENTS', 'MUNICIPAL COSTS', 
                  'OGD COSTS', 'INSURANCE PAYMENTS', 'NGO PAYMENTS']
df['TOTAL FINANCIAL SUPPORT'] = df[financial_cols].sum(axis=1)


df['COST PER PERSON'] = df['ESTIMATED TOTAL COST'] / (df['UTILITY - PEOPLE AFFECTED'] + 1e-5)


