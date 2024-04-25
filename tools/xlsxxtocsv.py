import pandas as pd

# 读取Excel文件
df = pd.read_excel(r"C:\Users\yegetables\PycharmProjects\pythonProject\input.xlsx")

# 将数据保存为CSV文件
df.to_csv('out.csv', index=False)