# 关于数据

## 1.公司要求版本(📂company)

**阈值设置**
- SIF：270
- SGF：250
- 异常值为700，**但数量较少几乎没有，所以没有单独筛选出来，在训练的时候需要单独处理**

**分层模式**

- 4268数据集完全嵌入Train训练集
- 其余数据集首先分成sif和sgf两种任务情况，按正负样本分层7：3抽取Train和Test
- csv中新增两列
    - **label：**在对应任务情况下已经二值化的结果
    - **source_name:**该数据来自于哪个数据集

**筛选条件**
- 【筛选单体】is_monomer=True
- 【除废数据】在对应任务（SIF/SGF）下，半衰期不为-1
- 【除废数据】SMILE不能为空

**每一种表征类型的数据结构**

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
└── README.md
