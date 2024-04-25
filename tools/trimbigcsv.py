import pandas as pd
import random
def trimbigcsv(input, out, n):
    # 读取包含1000万行数据的CSV文件
    df = pd.read_csv(input)

    # 随机抽取1000行数据
    sampled_df = df.sample(n=n)

    # 将抽取的数据保存为新的CSV文件
    sampled_df.to_csv(out, index=False)
import os
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)

input=current_dir+"\\..\\last\\loan\\data\\train.csv"
out=current_dir+"\\..\\last\\loan\\data\\train_part.csv"
trimbigcsv(input, out, 30000)
input=current_dir+"\\..\\last\\loan\\data\\testA.csv"
out=current_dir+"\\..\\last\\loan\\data\\testA_part.csv"
trimbigcsv(input, out, 6000)
# input="C:\\Users\\yegetables\\Downloads\\news_train_set.csv"
# out="C:\\Users\\yegetables\\PycharmProjects\\pythonProject\\news_part.csv"
# trimbigcsv(input,out,10000)

