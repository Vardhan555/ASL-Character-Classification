<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">American Sign Language (ASL) Classification - STRAWHATS</h3>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#experiments">Experiments</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
      </ul>
    </li>
    <li><a href="#test-phase">Test Phase</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Introduction
This initiative involves the creation of a machine learning model designed to categorize hand symbols in American Sign Language (ASL). The primary objective is to enhance communication for individuals who rely on ASL, thereby providing a valuable tool for improved accessibility.

### Dataset
The students in the course worked together to collectively generate a dataset for this project. This dataset comprises nine classes, with each student contributing 90 images, resulting in a total of 8,443 photos. These images depict hand gestures corresponding to ASL letters A through I, creating a diverse and extensive dataset. Through the implementation of this project, we had the opportunity to apply concepts learned in class to a real-world scenario, fostering practical experimentation and learning.


### Experiments
The project progressed through several phases, commencing with the establishment of a baseline through a thorough examination of existing research. Subsequently, the team engaged in data manipulation, augmenting the dataset by employing various image transformations like resizing, flipping, and rotation. Multiple experiments were then carried out, involving tasks such as dataset splitting, the incorporation of pre-trained classifiers (e.g., ResNet152), and the assessment of the model's performance.

---

&nbsp;

&nbsp;

<!-- GETTING STARTED -->
## Getting Started


### Using UF HiperGator:

1. To clone the private repository in the TERMINAL, execute the following code:
  ```sh
   git clone https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-strawhats.git

   ```
2. Navigate to "final-project-code-report-strawhats" folder and go to train.ipynb
    
4. Setup HyperGator
`Kernel: PyTorch-2.0.1 on HiperGator
Libraries: pandas, numpy, PIL, torch, torchvision, tqdm, Scikit-learn, Matplotlib, itertools, Scipy Optional, but recommended: GPU support for training

5. Execute the code
  
  
### Dependencies 
- Everything is included in kernel PyTorch-2.0.1 on HiperGator

---

&nbsp;

&nbsp;

<!-- Required Modifications -->

### Revise the file paths for training and label data, and designate locations for saving the final model using the following code:



- Modification required, input your own path for training and label data

```sh
# Paths to the data and labels stored in .npy format
train_data_file = 'data_train.npy'
train_label_file = 'labels_train.npy'
model_save_path = define_model_save_path('model/') # model will be saved here in the name of resnet_model_final.pth
.....
  ```

# Model
Leveraging the ResNet architecture, a Convolutional Neural Network (CNN) has been employed due to its effectiveness in image recognition tasks.


---

&nbsp;

&nbsp;

## Train phase
- Open the train.ipynb file.
- Load the dataset using the provided paths:
```python
  train_data_file = 'data_train.npy'
  train_label_file = 'labels_train.npy'
 ```
- Implement necessary transformations such as resizing, flipping, and rotation.
- Divide the dataset into training and testing sets.
- Before commencing training, ensure to specify the file path for saving the model for future use. Adjust the path in the code:
```python
 model_save_path = define_model_save_path('model/')
```
- To initiate the training process, go to Kernel -> Restart Kernel and Run All Cells.
## Test phase

### Load the saved ResNet152 model (resnet_model_final.pth) model from training phase

- Modifcation required, input best epoch from training phase

- Open the test.ipynb file.
- Provide paths for the testing data:
```python
train_data_file =  'data-1.npy'
train_label_file = 'labels-1.npy'
```
Specify the path to the saved model:
```python
model_file_path = 'model/resnet_model_final.pth'
```
---

## Model Evaluation

The model's performance is assessed using various metrics, including accuracy, F1 score, precision, and recall.

Test Loss: 0.0002, Test Accuracy: 99.26%

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A     | 1.00      | 1.00   | 1.00     | 30      |
| B     | 1.00      | 1.00   | 1.00     | 30      |
| C     | 1.00      | 0.97   | 0.98     | 30      |
| D     | 1.00      | 1.00   | 1.00     | 30      |
| E     | 1.00      | 1.00   | 1.00     | 30      |
| F     | 1.00      | 1.00   | 1.00     | 30      |
| G     | 0.94      | 1.00   | 0.97     | 30      |
| H     | 1.00      | 0.97   | 0.98     | 30      |
| I     | 1.00      | 1.00   | 1.00     | 30      |

Accuracy: 99.4%

Macro Average: Precision - 0.99, Recall - 0.99, F1-Score - 0.99

Weighted Average: Precision - 0.99, Recall - 0.99, F1-Score - 0.99

&nbsp;

&nbsp;

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

---

&nbsp;

&nbsp;

<!-- Authors -->
## Authors

#### Sai Vardhan Vegi - svegi@ufl.edu
#### Mithra Varun Bodanapati - mbodanapati@ufl.edu
#### Bhanu Prakash Reddy Vangala - bvangala1@ufl.edu

---
&nbsp;

&nbsp;

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

You can acknowledge any individual, group, institution or service.
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)

## Thank you

