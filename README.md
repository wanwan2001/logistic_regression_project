# 从零实现逻辑回归分类器

本项目旨在深入理解逻辑回归（Logistic Regression）分类算法的内部原理。我们分别使用了两种方法来解决一个经典的二元分类问题——威斯consin乳腺癌诊断预测：
1.  **Scikit-learn**: 使用业界标准的机器学习库快速搭建一个高性能的基线模型。
2.  **NumPy from Scratch**: 仅使用 NumPy 库，从零开始手动实现逻辑回归的核心算法，包括梯度下降。

---

## 🛠️ 技术栈 (Tech Stack)

* Python 3.12
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📂 项目结构 (Project Structure)

```
.
├── data
│   ├── raw/breast_cancer_data.csv  # 原始数据集
│   └── processed/                  # (本项目中未使用)
├── notebooks
│   ├── 01_initial_data_exploration.ipynb # 探索性数据分析 (EDA)
│   ├── 02_sklearn_logistic_regression.ipynb # Scikit-learn 实现
│   └── 03_numpy_logistic_regression_scratch.ipynb # NumPy 从零实现
├── src
│   ├── get_data.py                 # 获取并保存数据的脚本
│   └── __init__.py
├── .gitignore                      # Git 忽略文件配置
├── requirements.txt                # 项目依赖库
└── README.md                       # 项目说明文档
```

---

## 🚀 如何运行 (How to Run)

### 1. 环境设置

本项目在 **Python 3.12** 环境下开发和测试。

a. **克隆仓库** (上传后，这里的链接需要替换成你自己的)
   ```bash
   git clone [https://github.com/wanwan2001/logistic_regression_project.git]
   cd [logistic_regression_project]
   ```

b. **创建并激活 Conda 虚拟环境**
   ```bash
   conda create --name logistic_enve python=3.12
   conda activate logistic_enve
   ```

c. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 2. 使用说明

你可以按照以下顺序依次运行 `notebooks` 文件夹中的 Jupyter Notebooks，以完整地体验整个项目流程：
1.  **`01_initial_data_exploration.ipynb`**: 查看数据的基本情况和可视化。
2.  **`02_sklearn_logistic_regression.ipynb`**: 体验如何用 Scikit-learn 快速建模。
3.  **`03_numpy_logistic_regression_scratch.ipynb`**: 深入理解算法的内部实现。
4. 两个info文件分别展示了logistic回归的数学模型以及需要用到sklearn库的一些操作及功能介绍
---

## 📊 成果与结论 (Results)

通过两种不同的方法，我们都成功地构建了能够区分良性与恶性肿瘤的分类器。最终在测试集上的准确率如下：

| 实现方法 | 测试集准确率 (Accuracy) |
| Scikit-learn | **98.25%** |
| NumPy from Scratch | **98.25%** |

**结论**: 手动实现的逻辑回归模型，在经过适当的超参数调整后，其性能表现可以与专业的 `scikit-learn` 库相媲美。这有力地证明了对算法核心原理的理解是准确和有效的。
