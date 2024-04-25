import pandas as pd
import re
out="C:\\Users\\yegetables\\PycharmProjects\\pythonProject\\loan\\data\\train_part.csv"
input="C:\\Users\\yegetables\\PycharmProjects\\pythonProject\\loan\\data\\train.csv"
df = pd.read_csv(input)

# 仅保留isDefault列
isDefault_col = [col for col in df.columns if re.search(r'isDefault', col, re.IGNORECASE)]
df = df[isDefault_col]
df['isDefault'] = df['isDefault'].astype('int64')
# 打印包含isDefault列的DataFrame
print(df)

# pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer