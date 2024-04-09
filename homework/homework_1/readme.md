
# Binary Decision Tree Implementation and Usage

## Overview
This program demonstrates the workflow of using a custom implementation of a Binary Decision Tree for classification purposes. It includes functionalities such as loading a dataset from a CSV file, training the model, evaluating its accuracy, and visualizing the trained tree structure.

## Dataset Description
The dataset, "Diagnosis_7features.csv", contains 640 entries and 8 columns. Each entry represents a medical diagnosis record with 7 features, and each feature is a floating-point number. The last column is an integer representing the class label with a binary value indicating the outcome of the diagnosis.

### Features
- `p_bmi`: The patient's BMI (Body Mass Index).
- `personal_Hypertension`: Indicates if the patient has a history of hypertension.
- `personal_Hypertension_Year`: The number of years since the patient was diagnosed with hypertension.
- `personal_CHF`: Indicates if the patient has a history of congestive heart failure.
- `personal_PepticUlcer`: Indicates if the patient has a history of peptic ulcer disease.
- `SBP_pre`: The systolic blood pressure measurement taken prior to the diagnosis.
- `eGFR_pre`: The estimated glomerular filtration rate before diagnosis.



### Label
- `class`: The label indicating the diagnosis outcome (1 for positive, 0 for negative).

### First Few Entries
The first few entries of the dataset are as follows:
|   p_bmi |   personal_Hypertension |   personal_Hypertension_Year |   personal_CHF |   personal_PepticUlcer |   SBP_pre |   eGFR_pre |   class |
|--------:|------------------------:|-----------------------------:|---------------:|-----------------------:|----------:|-----------:|--------:|
|   0.722 |                   0.192 |                       -0.957 |         -0.106 |                 10.404 |    -0.609 |      0.652 |       1 |
|  -1.518 |                   0.192 |                       -0.957 |         -0.106 |                 -0.096 |    -1.07  |      1.047 |       1 |
|  -1.07  |                   0.192 |                       -0.54  |         -0.106 |                 -0.096 |    -1.809 |      0.215 |       1 |
|  -0.648 |                   0.192 |                       -0.123 |         -0.106 |                 10.404 |     0.592 |     -0.536 |       1 |
|  -0.666 |                   0.192 |                       -0.123 |         -0.106 |                 -0.096 |    -0.516 |     -0.085 |       1 |

## Key Features
- **Data Loading**: Load your dataset from a CSV file named "Diagnosis_7features.csv".
- **Model Training**: Train a decision tree model using the loaded dataset.
- **Accuracy Evaluation**: Calculate and display the accuracy of the trained model.
- **Tree Visualization**: Generate a DOT file to visualize the tree structure using Graphviz.

## Usage
1. Ensure the dataset file "Diagnosis_7features.csv" is located in the same directory as the executable.
2. Compile and run the program. It will display the loaded data, the model's accuracy, and generate a "tree.dot" file.
3. Optionally, use Graphviz to convert the "tree.dot" file into a graphical representation of the decision tree, or use an online Graphviz tool such as [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline).

## Model Accuracy
The accuracy of the trained Binary Decision Tree model is 0.720313.

## Tree Structure Visualization
Below is the visualized structure of the trained Binary Decision Tree.

![Binary Decision Tree Structure](result.png)

## Requirements
- A C++ compiler supporting C++11 or later.
- A dataset in a CSV file named "Diagnosis_7features.csv".
- Graphviz installed (optional) for generating a visual representation of the decision tree, or use an online Graphviz tool such as [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline).

## License
Copyright (c) 2024 by Chen-Yu Chang

