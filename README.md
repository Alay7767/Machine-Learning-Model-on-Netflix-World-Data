# Netflix Data Machine Learning Model


🧠 Project Overview

This project applies machine learning classification techniques to analyze and predict content attributes in the Netflix dataset.
The primary objective is to build models that can classify whether a title on Netflix is a Movie or a TV Show based on metadata such as genre, country, release year, and other available features.


Two models are trained and compared: Random Forest Classifier, Logistic Regression

The models are evaluated on accuracy and performance metrics such as confusion matrix and classification report.


📊 Dataset

The project uses a Netflix titles dataset, typically containing the following columns:

Column Name	Description
show_id	Unique ID for each show
type	Indicates whether the title is a Movie or TV Show
title	Title of the show
director	Director of the show (if available)
cast	Main cast members
country	Country of origin
date_added	Date added to Netflix
release_year	Year the title was released
rating	Age rating
duration	Length of the movie/show
listed_in	Genres or categories
description	Brief summary of the title

The target variable for this model is type (Movie vs. TV Show).


🧩 Features

Data Preprocessing

Missing value handling

Encoding categorical features with LabelEncoder

Scaling numerical features using StandardScaler

Model Training

Train-test split with train_test_split

Model training using RandomForestClassifier and LogisticRegression

Evaluation

Model accuracy score

Classification report

Confusion matrix visualization

Visualization

Exploratory Data Analysis (EDA) with matplotlib and seaborn


⚙️ Installation & Setup

1️⃣ Clone the repository
git clone [https://github.com/your-username/Netflix_Data_ML_Model.git](https://github.com/Alay7767/Machine-Learning-Model-on-Netflix-World-Data.git)
cd Netflix_Data_ML_Model

2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # For macOS/Linux
venv\Scripts\activate       # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Jupyter notebook
jupyter notebook Netflix_Data_ML_Model.ipynb

📦 Dependencies

Add these to requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn


🚀 Model Training & Evaluation

Both models are trained using the same preprocessed dataset and evaluated with standard metrics.

Example workflow:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


Metrics Used:

accuracy_score

classification_report

confusion_matrix

Example output:
Model	Accuracy	Notes
RandomForestClassifier	~95%	Performs better on non-linear relationships
LogisticRegression	~88%	Faster, interpretable baseline


📈 Visualization

The notebook also includes exploratory plots such as:

Distribution of content types

Content release trends over the years

Genre frequency

Correlation heatmaps


🧭 Future Improvements

Implement hyperparameter tuning using GridSearchCV

Add cross-validation for model robustness

Experiment with neural networks or XGBoost

Integrate a recommendation system component

Deploy model using Flask or FastAPI
