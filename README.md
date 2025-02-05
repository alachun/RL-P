## Project Description

This project implements multiple algorithms, including RL-P, RL-P(no mask), A2C, and HAC. Below is the structure of the project and instructions for its use.

## Project Structure


```

RL-P/
│
├── code/                     # Source code folder
│   ├── RL-P/                 # RL-P code
│   │   ├── child_snack_gym.py
│   │   ├── child_snack_gym2.py
│   │   ├── citycar_gym.py
│   │   ├── citycar_gym2.py
|   │   ├── ...
│   │   ├── pddlgym_data/     # PDDL files used for registration
│   │   ├── main.py           # Main program, used to call environments for training
│   |   └── model_test.py     # Testing program
│   │
│   └── HAC1/                 # HAC code
│       └── ...               # HAC code files
│   
│
├── data/                     # Original PDDL files
│   └── ...                   # Original PDDL files
│
├── requirements.txt          # Project dependencies
└── log/                      # Folder for storing test results
```
 


## Environment Registration

Before executing the code, please register the pddlgym environment. The registration process is detailed at https://github.com/tomsilver/pddlgym.

Since pddlgym does not support conditional effects and action costs, this project has made some modifications to the original files. The PDDL files used for registration are located in the RL-P/pddlgym_data folder. The original PDDL files are stored in the data folder.

### Usage Instructions

child_snack_gym.py, child_snack_gym2.py, citycar_gym.py, and citycar_gym2.py are the gym environments for the corresponding problems of the RL-P algorithm. RL-P(no mask) uses the same gym environments.
childsnack_pddlgym2.py, chilsnack_pddlgym.py, citycar_pddlgym.py, and citycar_pddlgym2.py are the corresponding pddlgym environments for the problems.
In main.py, the corresponding environments are called for training, and in model_test.py, the model is tested. The test results are stored in the log folder.

### Install Dependencies

Use the following command to install the required dependencies for the project:
```python

pip install -r requirements.txt
```

### Training the Model

Run the main.py file to start training the corresponding environment. You can modify the code as needed to select different environments and algorithms.
### Testing the Model

After training is complete, you can use the model_test.py file to test the model.

### Notes

Ensure that the pddlgym environment is correctly registered before executing the code.

Adjust the parameters in the code as needed to suit different experimental requirements.

## Acknowledgments


Thank you for using this project. We hope it will be helpful for your research or work!
