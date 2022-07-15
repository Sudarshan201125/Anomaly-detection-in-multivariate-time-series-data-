import pandas as pd
  
url = "cpu.csv"
df = pd.read_csv(url)
  
  
df_s1 = df[:100000]

df_s1 = df_s1.drop(df_s1[(df_s1.cpu == 'cpu-total')].index)

#print(df_s1)
df_s1.to_csv('edited_train.csv')


#(df_s1.cpu == 'cpu-total') & (df_s1.role == 'coupa_db') &