# 테스트용
import pandas as pd
df = pd.DataFrame({"polarity":["Positive","Negative","neutral"]})
print(df.loc[df["polarity"]=="negative"])
print(df.loc[df["polarity"].str.lower()=="negative"])
