# Netflix Analytics & ML Dashboard

An end-to-end **Data Science and Machine Learning project** built on the Netflix Titles dataset. The project covers the complete analytics workflow — from data cleaning and exploratory analysis to recommendation systems, forecasting, classification, model evaluation, and an interactive business intelligence dashboard.

The project was developed as part of a **Data Science Internship at Auspify Technologies** and extends the original task-based workflow into a reusable ML analytics application.

---

## Project Overview

This project analyzes Netflix content to answer practical business and analytical questions such as:

* What types of content dominate Netflix?
* How has Netflix's content library changed over time?
* Which countries and genres contribute the most content?
* Which titles are similar to a selected title?
* What are the expected future content-release trends?
* Can a machine learning model distinguish between Movies and TV Shows?
* What business insights can be extracted from the dataset?

The project combines **Python, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit** into one end-to-end workflow.

---

## Key Features

### Data Processing

* Dataset validation and cleaning
* Missing-value handling
* `"Not Given"` value normalization
* Date parsing
* Duration extraction
* Feature preparation
* Clean dataset generation

### Exploratory Data Analysis

* Movie vs TV Show distribution
* Content by country
* Content by genre
* Rating distribution
* Release-year trends
* Content growth analysis
* Visualization of major Netflix content patterns

### Recommendation System

A content-based recommendation system using:

* TF-IDF vectorization
* Cosine similarity
* Genre
* Director
* Content type

The system generates the **Top-5 similar titles** for a selected Netflix title.

### Trend Forecasting

A Linear Regression model is used to analyze historical content-release trends and generate forecasts for future years.

The project includes:

* Historical trend analysis
* Regression-based forecasting
* Historical vs predicted visualization
* Forecast outputs

### Machine Learning Classification

The project evaluates multiple classification algorithms for predicting:

> **Movie vs TV Show**

Models include:

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation includes:

* Accuracy
* Classification report
* Confusion matrix
* Feature importance
* Model comparison

### Business Intelligence Dashboard

The final application brings the analytical components together into an interactive dashboard.

The dashboard provides:

* Netflix content overview
* EDA visualizations
* Trend analysis
* Forecasting results
* Classification results
* Recommendation functionality
* Business-oriented insights

---

## Project Architecture

```text
Netflix-Analytics-ML-Dashboard/
│
├── .agents/
│   └── skills/
│
├── data/
│   ├── netflix_dataset.csv
│   └── netflix_cleaned.csv
│
├── models/
│   └── Saved ML models and preprocessing artifacts
│
├── outputs/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   ├── task5/
│   └── task6/
│
├── tasks/
│   ├── task1_data_cleaning.py
│   ├── task2_eda.py
│   ├── task3_recommendation_system.py
│   ├── task4_trend_prediction.py
│   ├── task5_classification_model.py
│   └── task6_business_dashboard.py
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technology Stack

| Category          | Technologies                                      |
| ----------------- | ------------------------------------------------- |
| Programming       | Python                                            |
| Data Processing   | Pandas, NumPy                                     |
| Visualization     | Matplotlib, Seaborn                               |
| Machine Learning  | Scikit-learn                                      |
| Recommendation    | TF-IDF, Cosine Similarity                         |
| Forecasting       | Linear Regression                                 |
| Classification    | Logistic Regression, Decision Tree, Random Forest |
| Dashboard         | Streamlit                                         |
| Model Persistence | Joblib / Pickle                                   |
| Development       | VS Code, Jupyter, Git, GitHub                     |

---

## Task Breakdown

| Task | Area                      | Techniques                                        |
| ---- | ------------------------- | ------------------------------------------------- |
| 1    | Data Cleaning             | Pandas, preprocessing                             |
| 2    | Exploratory Data Analysis | Statistical analysis, visualization               |
| 3    | Recommendation System     | TF-IDF, cosine similarity                         |
| 4    | Trend Prediction          | Linear Regression                                 |
| 5    | Classification            | Logistic Regression, Decision Tree, Random Forest |
| 6    | Business Dashboard        | EDA, forecasting, classification                  |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/talhaakbar1036/Netflix-Analytics-ML-Dashboard.git
cd Netflix-Analytics-ML-Dashboard
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run the complete task pipeline

```bash
python3 main.py
```

### Run selected tasks

For example:

```bash
python3 main.py --tasks 2 5
```

### Run an individual task

Example:

```bash
python3 tasks/task2_eda.py
```

### Launch the interactive dashboard

```bash
streamlit run app.py
```

The Streamlit application will provide an interactive interface for exploring the Netflix analytics and ML results.

---

## Machine Learning Workflow

```text
Netflix Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ├──────────────► Recommendation System
       │
       ├──────────────► Trend Forecasting
       │
       └──────────────► Classification
                              │
                              ▼
                       Model Evaluation
                              │
                              ▼
                    Saved Models / Results
                              │
                              ▼
                    Interactive Dashboard
```

---

## Recommendation System

The recommendation engine follows a content-based approach.

### Workflow

```text
Netflix Metadata
      │
      ▼
Feature Combination
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Cosine Similarity
      │
      ▼
Similarity Ranking
      │
      ▼
Top-5 Recommendations
```

The recommendation system uses available metadata such as genre, director, and content type to identify titles with similar characteristics.

---

## Classification

The classification component compares three machine learning algorithms:

```text
Logistic Regression
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        └── F1 Score

Decision Tree
        │
        └── Model Evaluation

Random Forest
        │
        └── Feature Importance
```

### Important Modeling Consideration

The classification model can achieve very high accuracy when `duration_value` is included because the original Netflix dataset represents duration differently for the two target classes:

* Movies → duration in minutes
* TV Shows → duration in seasons

Therefore, `duration_value` behaves as a near-leaky feature for predicting the target.

The project explicitly documents this limitation rather than presenting the resulting high accuracy as evidence of a production-ready classifier.

This is an important consideration when interpreting the model results.

---

## Forecasting

The forecasting component uses historical Netflix content-release data to estimate future trends.

The workflow includes:

1. Aggregating releases by year
2. Preparing the regression features
3. Training a Linear Regression model
4. Generating future predictions
5. Comparing historical and forecasted values
6. Visualizing the trend

---

## Outputs

Generated analytical outputs are organized under:

```text
outputs/
├── task1/
├── task2/
├── task3/
├── task4/
├── task5/
└── task6/
```

These may include:

* Text reports
* Recommendation results
* Classification reports
* Confusion matrices
* Feature-importance charts
* Model-comparison charts
* Forecast visualizations
* Business insight reports

---

## Dataset

The project uses the **Netflix Titles dataset**, containing information about Netflix movies and TV shows including:

* Title
* Type
* Director
* Cast
* Country
* Date Added
* Release Year
* Rating
* Duration
* Listed Genres
* Description

The dataset contains approximately **8,790 Netflix titles** after preprocessing.

---

## Business Insights

The project demonstrates how Netflix catalog data can be transformed into actionable analytical insights.

Potential business applications include:

* Content acquisition strategy
* Genre planning
* Regional content analysis
* Catalog growth monitoring
* Content recommendation
* Future content planning
* Audience-oriented content analysis
* Data-driven decision support

---

## Project Goals

This project was designed to demonstrate practical understanding of:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Natural-language feature representation
* Recommendation systems
* Supervised machine learning
* Model evaluation
* Forecasting
* Data visualization
* Dashboard development
* Business analytics
* End-to-end ML project organization

---

## Future Improvements

Potential improvements include:

* Advanced recommendation models
* Collaborative filtering
* Hybrid recommendation systems
* XGBoost-based classification
* More robust time-series forecasting
* Hyperparameter optimization
* Cross-validation
* Explainable AI
* Automated model retraining
* Docker deployment
* Cloud deployment
* MLflow experiment tracking
* CI/CD pipeline
* Production-grade data pipeline

---

## Limitations

This project is primarily an analytical and educational ML application.

Important limitations include:

* The dataset represents a snapshot of Netflix's catalog rather than live Netflix data.
* Forecasting is based on historical patterns and should not be interpreted as an actual Netflix business forecast.
* The Movie vs TV Show classifier is affected by the duration representation in the original dataset.
* The recommendation system is content-based and does not incorporate individual user behavior.
* Results depend on the quality and completeness of the source dataset.

---

## Author

**Talha Akbar**

Data Science | Machine Learning | Python | Analytics

GitHub:
https://github.com/talhaakbar1036

Portfolio:
https://italhaakbar.netlify.app/

LinkedIn:
https://www.linkedin.com/in/italhaakbar

---

## License

This project is licensed under the **MIT License**.
