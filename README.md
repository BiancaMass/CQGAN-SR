# PQWGAN - Edited
This repository contains the PQWGAN code, originally designed and made available at [PQWGAN GitHub directory](https://github.com/jasontslxd/PQWGAN).
This is the code of the paper _Hybrid Quantum-Classical Generative
Adversarial Network for High Resolution Image Generation_, Tsang et al., 2023.

I changed the code by adding documentation and using it as a starting point to develop code
for experiments related to my Master Thesis in Quantum GANs, for Maastricht University,
Master in Data Science for Decision Making.

## TODOs
1. Verify proper referencing of the paper.

## Requirements and Installation
We recommend working within a virtual environment. \
Please refer to requirements.txt for required libraries and version number.

**Note to self**: venv I have been using is venv1 (python 3.10) \

To run with the values of the basic experiment (binary MNIST) with default values from the 
original paper run the following command (note: hyperparameters values are hard-coded in the 
functions, refer to the code):\
`python train.py -cl "01" -d "mnist" -p 28 -l 8 -q 7 -b 25 -o "./output/231023_1953" -c 0 -ps 
1 28 `

Note: Classes can be set when calling the program, and it will automatically select a subset of 
the data with the right classes.

Where

| Input     | Description                             |
|-----------|-----------------------------------------|
| -cl       | `string`: name of the classes in training set e.g., "01" will be digits 0 and 1 in MNIST |
| -d        | `string`: training dataset (see folder structure) |
| -p        | `int`: number of patches to divide the images for training |
| -l        | `int`: number of layers |
| -q        | `int`: number of qubits (excl. ancilla) |
| -b        | `int`: batch size |
| -o        | `string`: destination dir for output |
| -c        | `bool`: whether to have  checkpoint|
| -rn       | `bool` : if True, draw latent vector from normal distribution. Else, from uniform. |
| -ps       | `int` `int`: shape of the image patch (for QG) |

Note: the output folder will be automatically named as: 
NumberOfClasses_NumberofPatches_NumberOfLayers_BatchSize, according to the provided parameters. 
If specified in the parameters, randn will be added, as well as patch shape.



## Contribution

To contribute to this project, please ensure that you follow the standard Python documentation guidelines:

- Write a descriptive docstring for each function, including:
   - Purpose:       Describe what the function does and how it does it.
   - Parameters:    List and describe the function's parameters.  Include their purpose, data types, and any default values they may have. 
   - Return Value:  Specify the value or values returned by the function, including data type. 
   - Usage Examples (opt.): Provide one or more examples demonstrating how to use the function correctly.
- When calling a function, always specify the arguments with appropriate descriptions.

### Commit message guidelines
Please adhere to the following guidelines for commit messages:

| Keyword  | Description                                                                                                                         |
|----------|-------------------------------------------------------------------------------------------------------------------------------------|
| feat     | Denotes a new feature or enhancement added to the codebase.                                                                         |
| fix      | Indicates a bug fix or resolution to an issue.                                                                                      |
| refactor | Denotes code refactoring, which involves restructuring or optimizing existing code without changing its functionality.                |
| docs     | Indicates changes or additions to documentation.                                                                                    |
| chore    | Refers to changes in build tasks, dependencies, or other maintenance-related tasks that don't affect the code's functionality.      |
| style    | Denotes code style changes, such as formatting, indentation, or whitespace modifications.                                           |
| test     | Indicates changes or additions to tests or testing-related code.                                                                    |
| perf     | Denotes performance-related improvements or optimizations.                                                                          |
| revert   | Indicates a commit that reverts a previous commit.                                                                                   |
| merge    | Refers to merge commits when combining branches.                                                                                     |


## Citations and Acknowledgements