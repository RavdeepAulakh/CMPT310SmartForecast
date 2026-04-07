# CMPT310 Smart Forecast

A machine learning project for predicting exam scores using various regression models.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CMPT310SmartForecast
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the complete analysis pipeline:

```bash
python main.py
```

This will:
- Load and preprocess the exam data from `data/exams.csv`
- Perform exploratory data analysis (EDA) with visualizations
- Train baseline, Random Forest, and SVR models
- Compare model performance
- Evaluate the best performing model with predictions and residual plots

## Project Structure

```
.
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── data/
│   └── exams.csv       # Exam dataset
├── src/
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── eda.py                 # Exploratory data analysis
│   ├── baseline_model.py      # Baseline model training
│   ├── model_experiments.py   # Random Forest and SVR models
│   └── final_model.py         # Best model selection and evaluation
└── README.md           # This file
```

## Data

The project uses `data/exams.csv` which contains exam-related data for the Smart Forecast prediction task.

## License

We allow our project to be shared or used in future CMPT310 Class Sessions
