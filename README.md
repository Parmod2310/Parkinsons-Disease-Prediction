# Parkinsons-Disease-Prediction

Developed at Acmegrade, this project focuses on predicting Parkinson’s disease using machine learning. The aim is to analyze voice data to predict the likelihood of Parkinson’s, aiding in early diagnosis for more effective treatment.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The objective of this project is to build a machine learning model to classify if an individual has Parkinson’s disease based on voice measurements. This model is trained to detect patterns in vocal changes that may indicate the presence of Parkinson's disease.

## Dataset
The dataset, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons), consists of 24 biomedical voice measurements from individuals with and without Parkinson’s disease. Key features include:
- **MDVP: Fo(Hz)** – Average vocal fundamental frequency
- **MDVP: Jitter(%)** – Variation in fundamental frequency
- **MDVP: Shimmer** – Variation in amplitude
- **NHR, HNR** – Signal-to-noise measures, among others

## Installation
Clone this repository and install the required packages to run the project.

```bash
git clone https://github.com/yourusername/PRJ-Parkinson-Disease-Prediction.git
cd PRJ-Parkinson-Disease-Prediction
```

```bash
pip install -r requirements.txt
 ```

## Requirements
- **Python 3.x**
- **Libraries**: numpy, pandas, scikit-learn, matplotlib, seaborn, ydata_profiling, jupyter
  
## Usage
1. Launch the Jupyter notebook:
```bash
jupyter notebook PRJ-Parkinson\ Disease\ Prediction.ipynb
```
2. Run each cell to:
- Load and analyze the dataset
- Preprocess and clean data
- Train and evaluate models
3. Review model metrics to identify the best-performing algorithm.
## Model Performance
This project tests several models, including:

- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
Model evaluation includes accuracy, precision, recall, and F1-score to determine the best-performing classifier.

## Technologies Used
The project utilizes a variety of technologies and machine learning algorithms for predictive analysis:

- **Programming Languages**: Python
- **Data Analysis Libraries**: Pandas, NumPy
- **Machine Learning Libraries**: Scikit-Learn
- **Visualization Libraries**: Matplotlib, Seaborn
- **Exploratory Data Analysis (EDA)**: ydata_profiling

- **Machine Learning Algorithms**:

     - **Logistic Regression**: A classification algorithm useful for binary classification tasks.
     - **Support Vector Machines (SVM)**: Effective in high-dimensional spaces, used for classification.
     - **Random Forest**: An ensemble method that creates a forest of decision trees, improving accuracy.
     - **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that predicts based on proximity in feature space.
## Contributing
Contributions are encouraged! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or additional information, feel free to reach out:

- **Name:** Parmod
- **Email:** p921035@gmail.com
- **LinkedIn:** [Parmod's LinkedIn Profile](https://www.linkedin.com/in/parmod2310/)

### This README provides a complete overview of the project, from installation and usage to technologies and contributing.

Feel **free to** replace **any** placeholders **with** the appropriate **values** before adding this file **to** your repository!
