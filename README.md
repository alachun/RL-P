# RL-P

## 项目说明

本项目实现了多种算法，包括RL-P、RL-P(no mask)、A2C和HAC，主要针对特定的环境进行训练和测试。以下是该项目的结构及使用说明。

## 项目结构


```

RL-P/
│
├── code/                     # 源代码文件夹
│   ├── RL-P/                 # RL-P相关代码
│   │   ├── child_snack_gym.py
│   │   ├── child_snack_gym2.py
│   │   ├── citycar_gym.py
│   │   ├── citycar_gym2.py
|   │   ├── ...
│   │   ├── pddlgym_data/     # 注册时使用的PDDL文件
│   │   ├── main.py           # 主程序，调用环境进行训练
│   |   └── model_test.py     # 测试程序
│   │
│   └── HAC1/                 # HAC相关代码
│       └── ...               # HAC相关代码文件
│   
│
├── data/                     # 原始PDDL文件
│   └── ...                   # 原始PDDL文件
│
├── requirements.txt          # 项目依赖库
└── log/                      # 测试结果存放文件夹
```
 


## 环境注册

在执行代码之前，请先注册 pddlgym 环境。注册流程详见  https://github.com/tomsilver/pddlgym。

由于 pddlgym 不支持条件效果和动作成本，本项目对原文件做了一些删改，注册时使用的 PDDL 文件位于 RL-P/pddlgym_data 文件夹内。原 PDDL 文件位于 data 文件夹中。

### 使用说明

child_snack_gym.py，child_snack_gym2.py，citycar_gym.py，citycar_gym2.py是RL-P算法相应问题的gym环境，RL-P(no mask)使用同样的gym环境。childsnack_pddlgym2.py，chilsnack_pddlgym.py，citycar_pddlgym.py，citycar_pddlgym2.py为相应问题的pddlgym环境。在main.py中调用相应问题的环境进行训练，在model_test.py中进行测试。测试结果位于log文件夹中。

### 安装依赖

使用以下命令安装项目所需的依赖库：
```python

pip install -r requirements.txt
```

### 训练模型

运行 main.py 文件以开始训练相应问题的环境。可以根据需要修改代码以选择不同的环境和算法。

### 测试模型

训练完成后，可以使用 model_test.py 文件进行模型测试。

### 注意事项

请确保在执行代码之前，已正确注册 pddlgym 环境。
根据需要调整代码中的参数，以适应不同的实验需求。

## 结语


感谢您使用本项目，希望本项目能对您的研究或工作有所帮助！
