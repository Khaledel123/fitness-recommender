# fitness-recommender
A machine learning project that recommends personalized workout plans by implementing and comparing three recommendation approaches: Content-Based Filtering, Collaborative Filtering (Matrix Factorization), and Neural Collaborative Filtering.
# FitRec: A Multi-Approach Fitness Workout Recommender

A machine learning project that recommends personalized workout plans by implementing and comparing three recommendation approaches: Content-Based Filtering, Collaborative Filtering (Matrix Factorization), and Neural Collaborative Filtering.

## Project Overview

**Problem:** Given a user's fitness profile (goals, fitness level, available equipment, time constraints), recommend a personalized set of exercises to form a workout plan.

**What makes this project stand out:**
- Implements 3 distinct ML approaches on the same problem
- Rigorous evaluation and comparison using ranking metrics
- Synthetic user-interaction data generation with documented methodology
- End-to-end pipeline from data processing to model evaluation
- Clean, modular, well-documented code

---

## Tools & Setup

### Language
- **Python 3.10+** (primary language for ML/data science ecosystem)

### Recommended IDE
- **VS Code** with the following extensions:
  - Python (Microsoft)
  - Jupyter (Microsoft)
  - Pylance (for type checking)
  - GitLens (for version control)
- Alternative: **PyCharm** (Community Edition is free)

### Core Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| **pandas** | Data manipulation & cleaning | `pip install pandas` |
| **numpy** | Numerical computing | `pip install numpy` |
| **scikit-learn** | ML utilities, preprocessing, metrics | `pip install scikit-learn` |
| **PyTorch** | Neural Collaborative Filtering model | `pip install torch` |
| **surprise** | Collaborative filtering (SVD/ALS) | `pip install scikit-surprise` |
| **matplotlib** | Plotting & visualization | `pip install matplotlib` |
| **seaborn** | Statistical visualizations | `pip install seaborn` |
| **jupyter** | Interactive notebooks for EDA | `pip install jupyter` |
| **streamlit** | (Optional) Interactive demo app | `pip install streamlit` |

### Quick Install
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Dataset
- **Primary:** [The Ultimate Gym Exercises Dataset](https://www.kaggle.com/datasets/peshimaammuzammil/the-ultimate-gym-exercises-dataset-for-all-levels) (~2,900 exercises)
  - Columns: Exercise Name, Description, Type, Body Part, Equipment, Difficulty Level, Rating
- **Supplementary:** Synthetic user profiles & interaction data (generated in `src/data_generation.py`)

### Version Control
- **Git + GitHub** — push your code to a public repo for your application
- Write clear commit messages showing your development process

---

## Project Structure

```
fitness-recommender/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                           # Original dataset (don't modify)
│   │   └── gym_exercises.csv
│   ├── processed/                     # Cleaned & feature-engineered data
│   │   ├── exercises_clean.csv
│   │   ├── user_profiles.csv
│   │   └── user_exercise_interactions.csv
│   └── README.md                      # Data dictionary & documentation
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_data_generation.ipynb       # Synthetic user data creation
│   ├── 03_content_based.ipynb         # Content-based model experiments
│   ├── 04_collaborative_filtering.ipynb  # SVD/ALS experiments
│   ├── 05_neural_cf.ipynb             # Neural model experiments
│   └── 06_evaluation_comparison.ipynb # Final comparison & analysis
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Data loading & cleaning
│   ├── data_generation.py             # Synthetic user/interaction generation
│   ├── feature_engineering.py         # Exercise & user feature vectors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── content_based.py           # Content-based filtering
│   │   ├── collaborative_filtering.py # Matrix factorization (SVD)
│   │   └── neural_cf.py              # Neural collaborative filtering
│   ├── evaluation.py                  # Metrics: Precision@K, NDCG, etc.
│   ├── recommender.py                 # Unified recommendation interface
│   └── utils.py                       # Helper functions
│
├── results/
│   ├── figures/                       # Generated plots & charts
│   └── metrics/                       # Saved evaluation results
│
├── app/                               # (Optional) Streamlit demo
│   └── streamlit_app.py
│
└── tests/                             # Unit tests (bonus points)
    └── test_models.py
```

---

## 🔬 Methodology

### Phase 1: Data Preparation (Days 1-2)
1. Download and clean the exercise dataset
2. Engineer exercise features (encode body part, equipment, difficulty, type)
3. Generate synthetic user profiles with realistic fitness goals
4. Simulate user-exercise interactions based on logical preference rules

### Phase 2: Model Implementation (Days 3-8)

#### Model 1: Content-Based Filtering
- Represent exercises as feature vectors (one-hot/multi-hot encoding of attributes)
- Create user preference profiles from their stated goals
- Rank exercises by cosine similarity between user profile and exercise vectors
- **Key concepts:** TF-IDF on descriptions, cosine similarity, feature weighting

#### Model 2: Collaborative Filtering (Matrix Factorization)
- Build user-exercise interaction matrix from synthetic data
- Apply SVD (Singular Value Decomposition) via Surprise library
- Learn latent factors that capture hidden user preferences
- **Key concepts:** Matrix factorization, latent factors, cold-start problem

#### Model 3: Neural Collaborative Filtering
- PyTorch model combining user & exercise embeddings
- MLP layers learn non-linear interaction patterns
- Can incorporate both collaborative signals AND content features
- **Key concepts:** Embeddings, multi-layer perceptron, BCE loss, implicit feedback

### Phase 3: Evaluation & Analysis (Days 9-11)

| Metric | What it measures |
|--------|------------------|
| **Precision@K** | Of the top K recommendations, how many are relevant? |
| **Recall@K** | Of all relevant exercises, how many appear in top K? |
| **NDCG@K** | Are the most relevant exercises ranked higher? |
| **Hit Rate** | Does at least one relevant exercise appear in top K? |
| **Coverage** | What fraction of exercises ever get recommended? |

### Phase 4: Polish & Documentation (Days 12-14)
- Write analysis comparing the three approaches
- Create visualizations (performance charts, embedding visualizations)
- Clean up code, add docstrings, write README
- (Optional) Build Streamlit demo app

---

 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fitness-recommender.git
cd fitness-recommender

# 2. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download dataset
# Place gym_exercises.csv in data/raw/

# 4. Run the pipeline
python src/data_processing.py        # Clean data
python src/data_generation.py        # Generate synthetic users
python src/models/content_based.py   # Train content-based model
python src/models/collaborative_filtering.py  # Train SVD model
python src/models/neural_cf.py       # Train neural model
python src/evaluation.py             # Compare all models

# 5. (Optional) Launch demo
streamlit run app/streamlit_app.py
```

---

## Results

> _To be filled in after running experiments_

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|-------------|-----------|---------|
| Content-Based | - | - | - |
| SVD (Collaborative) | - | - | - |
| Neural CF | - | - | - |

---

## Key Takeaways

> _To be filled in after analysis — this is the section admissions committees care about most!_

Discuss:
- Which approach works best and **why**
- Trade-offs between approaches (cold-start, scalability, interpretability)
- Limitations of synthetic data and how real data would change results
- What you'd do differently with more time/data

---

## Author
**Kelk** — Software Engineering, University of Texas at Dallas
