import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import json

st.set_page_config(page_title="FitRec 🏋️", page_icon="🏋️", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    exercises = pd.read_csv('data/processed/exercises_clean.csv')
    users = pd.read_csv('data/processed/user_profiles.csv')
    interactions = pd.read_csv('data/processed/user_exercise_interactions.csv')
    return exercises, users, interactions

exercises, users, interactions = load_data()

# ---- Content-Based Setup ----
@st.cache_data
def setup_content_based(exercises):
    feature_cols = ['body_part', 'equipment', 'movement_type', 'movement_pattern', 'difficulty']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    exercise_features = encoder.fit_transform(exercises[feature_cols])
    feature_names = encoder.get_feature_names_out(feature_cols)
    return exercise_features, feature_names

exercise_features, feature_names = setup_content_based(exercises)

def build_user_profile(user_input, feature_names):
    profile = np.zeros(len(feature_names))
    fname_list = list(feature_names)

    focus = user_input['preferred_body_focus']
    body_map = {
        'upper': ['chest', 'back', 'shoulders', 'arms'],
        'lower': ['legs'],
        'core': ['core'],
        'full': ['chest', 'back', 'shoulders', 'arms', 'legs', 'core'],
    }
    for bp in body_map.get(focus, []):
        col = f'body_part_{bp}'
        if col in fname_list:
            profile[fname_list.index(col)] = 1.0 if focus != 'full' else 0.6

    equip_map = {
        'bodyweight_only': {'bodyweight': 1.0, 'pull-up bar': 0.5},
        'dumbbells': {'bodyweight': 0.7, 'dumbbells': 1.0, 'pull-up bar': 0.3},
        'home_gym': {'bodyweight': 0.5, 'dumbbells': 0.8, 'barbell': 0.8, 'pull-up bar': 0.6},
        'full_gym': {'bodyweight': 0.4, 'dumbbells': 0.7, 'barbell': 0.9, 'machine': 1.0, 'pull-up bar': 0.6},
    }
    for equip, weight in equip_map.get(user_input['equipment_access'], {}).items():
        col = f'equipment_{equip}'
        if col in fname_list:
            profile[fname_list.index(col)] = weight

    diff_map = {
        'beginner': {'beginner': 1.0, 'intermediate': 0.3, 'advanced': 0.0},
        'intermediate': {'beginner': 0.2, 'intermediate': 1.0, 'advanced': 0.4},
        'advanced': {'beginner': 0.0, 'intermediate': 0.4, 'advanced': 1.0},
    }
    for diff, weight in diff_map.get(user_input['fitness_level'], {}).items():
        col = f'difficulty_{diff}'
        if col in fname_list:
            profile[fname_list.index(col)] = weight

    goal = user_input['fitness_goal']
    if goal in ['muscle_gain', 'weight_loss']:
        if 'movement_type_compound' in fname_list:
            profile[fname_list.index('movement_type_compound')] = 0.8
        if 'movement_type_isolation' in fname_list:
            profile[fname_list.index('movement_type_isolation')] = 0.5
    elif goal == 'endurance':
        if 'movement_type_compound' in fname_list:
            profile[fname_list.index('movement_type_compound')] = 0.6
        if 'movement_type_isolation' in fname_list:
            profile[fname_list.index('movement_type_isolation')] = 0.6
    elif goal == 'flexibility':
        if 'movement_type_isolation' in fname_list:
            profile[fname_list.index('movement_type_isolation')] = 0.8
    else:
        if 'movement_type_compound' in fname_list:
            profile[fname_list.index('movement_type_compound')] = 0.6
        if 'movement_type_isolation' in fname_list:
            profile[fname_list.index('movement_type_isolation')] = 0.5

    return profile

# ---- SVD Setup ----
@st.cache_data
def setup_svd(users, exercises, interactions):
    user_ids = sorted(users['user_id'].unique())
    exercise_ids = sorted(exercises['exercise_id'].unique())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    exercise_to_idx = {eid: i for i, eid in enumerate(exercise_ids)}

    R = np.zeros((len(user_ids), len(exercise_ids)))
    for _, row in interactions.iterrows():
        u = user_to_idx[row['user_id']]
        e = exercise_to_idx[row['exercise_id']]
        R[u, e] = row['rating']

    user_means = np.zeros(R.shape[0])
    R_centered = R.copy()
    for i in range(R.shape[0]):
        rated = R[i] > 0
        if rated.sum() > 0:
            user_means[i] = R[i, rated].mean()
            R_centered[i, rated] -= user_means[i]

    svd = TruncatedSVD(n_components=15, random_state=42)
    user_factors = svd.fit_transform(R_centered)
    R_pred = user_factors @ svd.components_
    for i in range(R_pred.shape[0]):
        R_pred[i] += user_means[i]

    return R_pred, user_to_idx, exercise_to_idx

R_pred, svd_user_to_idx, svd_exercise_to_idx = setup_svd(users, exercises, interactions)

# ---- Neural CF Setup ----
class NeuralCF(nn.Module):
    def __init__(self, n_users, n_exercises, n_factors=32, hidden_layers=[64, 32, 16], dropout=0.2):
        super().__init__()
        self.user_emb_gmf = nn.Embedding(n_users, n_factors)
        self.exercise_emb_gmf = nn.Embedding(n_exercises, n_factors)
        self.user_emb_mlp = nn.Embedding(n_users, n_factors)
        self.exercise_emb_mlp = nn.Embedding(n_exercises, n_factors)
        mlp_layers = []
        input_size = n_factors * 2
        for hidden_size in hidden_layers:
            mlp_layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout)])
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.predict = nn.Linear(n_factors + hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, exercise_ids):
        gmf_out = self.user_emb_gmf(user_ids) * self.exercise_emb_gmf(exercise_ids)
        mlp_input = torch.cat([self.user_emb_mlp(user_ids), self.exercise_emb_mlp(exercise_ids)], dim=-1)
        mlp_out = self.mlp(mlp_input)
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.sigmoid(self.predict(concat)).squeeze()

# =============================================
# UI
# =============================================

st.title("🏋️ FitRec — Fitness Workout Recommender")
st.markdown("*Tell us about yourself and we'll recommend exercises using 3 different ML models.*")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Profile")
    fitness_goal = st.selectbox("🎯 Fitness Goal", ['weight_loss', 'muscle_gain', 'endurance', 'flexibility', 'general_fitness'],
                                format_func=lambda x: x.replace('_', ' ').title())
    fitness_level = st.selectbox("📊 Fitness Level", ['beginner', 'intermediate', 'advanced'],
                                 format_func=lambda x: x.title())
    equipment = st.selectbox("🔧 Equipment Access", ['bodyweight_only', 'dumbbells', 'home_gym', 'full_gym'],
                             format_func=lambda x: x.replace('_', ' ').title())
    body_focus = st.selectbox("💪 Body Focus", ['full', 'upper', 'lower', 'core'],
                              format_func=lambda x: x.title())
    time_available = st.slider("⏱️ Time per Session (min)", 15, 90, 45, step=15)
    n_exercises = st.slider("📋 Number of Exercises", 3, 15, 8)

with col2:
    st.subheader("Your Workout Plan")

    if st.button("🚀 Get Recommendations!", type="primary", use_container_width=True):
        user_input = {
            'fitness_goal': fitness_goal,
            'fitness_level': fitness_level,
            'equipment_access': equipment,
            'preferred_body_focus': body_focus,
            'time_per_session': time_available,
        }

        # --- Content-Based ---
        profile = build_user_profile(user_input, feature_names)
        sims = cosine_similarity(profile.reshape(1, -1), exercise_features).flatten()
        cb_indices = np.argsort(sims)[::-1][:n_exercises]
        cb_recs = exercises.iloc[cb_indices][['exercise_name', 'body_part', 'equipment', 'difficulty', 'est_duration_min']].copy()
        cb_recs['score'] = sims[cb_indices]

        # --- SVD (find most similar existing user) ---
        # Match to closest user profile
        best_user = None
        best_match = -1
        for _, u in users.iterrows():
            match = sum([
                u['fitness_goal'] == fitness_goal,
                u['fitness_level'] == fitness_level,
                u['equipment_access'] == equipment,
                u['preferred_body_focus'] == body_focus,
            ])
            if match > best_match:
                best_match = match
                best_user = u['user_id']

        u_idx = svd_user_to_idx[best_user]
        pred_ratings = R_pred[u_idx]
        svd_indices = np.argsort(pred_ratings)[::-1][:n_exercises]
        svd_recs = exercises.iloc[svd_indices][['exercise_name', 'body_part', 'equipment', 'difficulty', 'est_duration_min']].copy()
        svd_recs['score'] = pred_ratings[svd_indices]

        # --- Display Results ---
        total_time_cb = cb_recs['est_duration_min'].sum()
        total_time_svd = svd_recs['est_duration_min'].sum()

        tab1, tab2 = st.tabs(["🧠 Content-Based", "📊 SVD Collaborative"])

        with tab1:
            st.markdown(f"**Estimated workout time: {total_time_cb:.0f} min**")
            for i, (_, row) in enumerate(cb_recs.iterrows(), 1):
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f"**{i}. {row['exercise_name'].title()}**")
                    c2.caption(f"🎯 {row['body_part'].title()}")
                    c3.caption(f"🔧 {row['equipment'].title()}")
                    c4.caption(f"{'🟢' if row['difficulty']=='beginner' else '🟡' if row['difficulty']=='intermediate' else '🔴'} {row['difficulty'].title()}")

        with tab2:
            st.markdown(f"**Estimated workout time: {total_time_svd:.0f} min**")
            for i, (_, row) in enumerate(svd_recs.iterrows(), 1):
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f"**{i}. {row['exercise_name'].title()}**")
                    c2.caption(f"🎯 {row['body_part'].title()}")
                    c3.caption(f"🔧 {row['equipment'].title()}")
                    c4.caption(f"{'🟢' if row['difficulty']=='beginner' else '🟡' if row['difficulty']=='intermediate' else '🔴'} {row['difficulty'].title()}")

# Sidebar with model comparison
with st.sidebar:
    st.header("📈 Model Performance")
    try:
        with open('results/metrics/content_based_results.json') as f:
            cb_res = json.load(f)
        with open('results/metrics/collaborative_filtering_results.json') as f:
            cf_res = json.load(f)
        with open('results/metrics/neural_cf_results.json') as f:
            ncf_res = json.load(f)

        comparison = pd.DataFrame({
            'Content-Based': cb_res,
            'SVD': cf_res,
            'Neural CF': ncf_res,
        }).T
        st.dataframe(comparison.style.highlight_max(axis=0), use_container_width=True)
    except:
        st.info("Run all notebooks first to see metrics.")

    st.divider()
    st.markdown("**Built with:** Python, PyTorch, scikit-learn, Streamlit")
    st.markdown("[GitHub Repo](https://github.com/Khaledel123/fitness-recommender)")
