# 关于数据【遇到问题和情况，请随时在群里at我】

### 0.目前已有的数据
- Avalon
- Morgan
- With(0):理化性质
- molFormer
- MolT5
- Morgan+Avalon
- Avalon+MolFormer
- Avalon+MolT5

其中，数目比例的情况为

    Train_sif_*         样本数: 398
    Test_sif_*          样本数: 117
    Train_sgf_*         样本数: 307
    Test_sgf_*          样本数: 95


### **1.阈值设置**
- SIF：270
- SGF：250
- 异常值为700，**但数量较少几乎没有，所以没有单独筛选出来，在训练的时候需要单独处理**

### **2.分层模式（公司模式和论文模式）**

#### 2.1 公司模式的划分方法（📂company）
- 4268数据集完全嵌入Train训练集
- 其余数据集首先分成sif和sgf两种任务情况，按正负样本分层7：3抽取Train和Test
- 【目标】：按照公司的要求，存入4268数据集并且分层


#### 2.2 论文模式的划分方法（📂paper）【暂时未更新】
- 将五种专利的数据集整合，以相似度为边，建立无向图
- 按照独立联通分量作为聚类
- 按照独立分量中点的数目，使用贪心算哒按照7：3划分出数据集
- 【目标】：保证Train和Test相似度尽可能低


### **3.已经完成的筛选和操作**
- 【筛选单体】is_monomer=True
- 【除废数据】在对应任务（SIF/SGF）下，半衰期不为-1
- 【除废数据】SMILE不能为空
- 【csv中新增两列】：
    - **label：**在对应任务情况下已经二值化的结果
    - **source_name:**该数据来自于哪个数据集

### **4.每一种表征类型的数据结构**

    .
    ├── 📂csv/                          # Processed annotations (CSV format)
    │   ├── Train_sif_{repr}.csv
    │   ├── Test_sif_{repr}.csv
    │   ├── Train_sgf_{repr}.csv
    │   └── Test_sgf_{repr}.csv
    │
    ├── 📂features/                     # Extracted molecular representations
    │   ├── Train_sif_{repr}.npz
    │   ├── Test_sif_{repr}.npz
    │   ├── Train_sgf_{repr}.npz
    │   └── Test_sgf_{repr}.npz
    │
    ├── 📂npy_data/                     # Final NumPy datasets for modeling
    │   ├── 📂SIF/
    │   │   ├── 📂Train/
    │   │   │   ├── x_train_sif.npy              # Feature matrix (model input)
    │   │   │   ├── y_train_sif.npy              # Continuous labels (minutes)
    │   │   │   ├── y_train_sif_label.npy        # Binarized labels
    │   │   │   └── train_sif_source_name.npy    # Data source identifiers
    │   │   └── 📂Test/
    │   │       ├── x_test_sif.npy
    │   │       ├── y_test_sif.npy
    │   │       ├── y_test_sif_label.npy
    │   │       └── test_sif_source_name.npy
    │   │
    │   ├── 📂SGF/
    │   │   ├── 📂Train/
    │   │   │   ├── x_train_sgf.npy
    │   │   │   ├── y_train_sgf.npy
    │   │   │   ├── y_train_sgf_label.npy
    │   │   │   └── train_sgf_source_name.npy
    │   │   └── 📂Test/
    │   │       ├── x_test_sgf.npy
    │   │       ├── y_test_sgf.npy
    │   │       ├── y_test_sgf_label.npy
    │   │       └── test_sgf_source_name.npy
    │   │
    │   ├── feature_names.npy          # Feature names (NumPy format)
    │   └── feature_names.json         # Feature names (JSON format)
    │
    ├── 🔢 split.py: 分割数据为npy的脚本【不影响工作流程】
    │
    └── README.md
