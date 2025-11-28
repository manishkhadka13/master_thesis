import pandas as pd


df1 = pd.read_csv("stratified_50_samples_per_model - stratified_50_samples_per_model.csv")
df2 = pd.read_csv("50_samples_falcon11b - 50_samples_falcon11b.csv")
df3=pd.read_csv("50_samples_llama13b - 50_samples_llama13b.csv")


merged_df = pd.concat([df1, df2,df3], ignore_index=True)


merged_df.to_csv("human_validation_data.csv", index=False)

