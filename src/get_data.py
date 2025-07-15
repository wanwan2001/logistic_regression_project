#从scikit-learn中获得数据集
import pandas as pd 
from sklearn.datasets import load_breast_cancer
import os 

# data = load_breast_cancer()
# print("数据特征",data.data.shape)
# print("qianjihang", data.data[:5])
# print("目标变量",data.target) # 这个数据集中的标签和数据是分开存储的。/

def fetch_and_save_data():
    # 创建一个函数用来保存和展示data，从 sklearn中加载数据集，保存为.csv位置在data/raw文件夹
    cancer_dataset = load_breast_cancer()
    # 处理成dataframe格式，更方便操作。
    
    x = pd.DataFrame(cancer_dataset.data,columns=cancer_dataset.feature_names) # save features matrix 
    y = pd.Series(cancer_dataset.target, name='target') # series 提取单独一列向量feature 
    
    # 拼接 x y,按列拼接
    df = pd.concat([x,y], axis=1) 
    
    # 创建保存路径，os.path.join()可以自动处理路径分割符，提高跨系统的兼容性。
    save_path = os.path.join('data','raw','breast_cancer_data.csv')
    
    # 确保目录存在 os.makedirs 创建目录，os.path.dirname()提取目录部分,exist_ok = True确保存在目录仍然可以继续进行，避免报错 
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    
    # 保存
    df.to_csv(save_path,index=False)
    
    # 函数成功指针
    print("success")
    print(f"dataset has {df.shape[0]} rows,{df.shape[1]} columns")
    print(f"file has saved {save_path}")
    
if __name__ == "__main__":
    fetch_and_save_data()