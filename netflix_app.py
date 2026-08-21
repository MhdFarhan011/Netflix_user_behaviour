import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Netflix Churn Insights", layout="wide")

NUM_COLS = ['age', 'account_age_months', 'session_count',
            'avg_watch_time_minutes_per_week', 'watch_sessions_per_week',
            'completion_rate', 'avg_rating_given', 'app_rating',
            'recommendation_click_rate', 'days_since_last_login']
CAT_COLS = ['gender', 'region', 'subscription_type', 'payment_method',
            'primary_device', 'favorite_genre', 'time_of_day', 'recommendation_source']


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df[NUM_COLS + CAT_COLS], columns=CAT_COLS, drop_first=True)
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Extract feature importances instead of linear coefficients
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, X.columns, auc, importance


def build_feature_row(input_dict, feature_columns):
    """Turn a single customer's raw inputs into a model-ready one-hot row."""
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row, columns=CAT_COLS)
    # align to training columns, fill missing dummy cols with 0
    row_final = row_encoded.reindex(columns=feature_columns, fill_value=0)
    return row_final


# ---------- Sidebar ----------
st.sidebar.title("📺 Churn Insights App")

DATA_PATH = "netflix_user_behavior_churn_50000.csv"
df = load_data(DATA_PATH)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Driver Analysis", "📥 Batch Scoring", "🎛️ What-If Simulator"]
)

# Unpack updated return elements cleanly (No scaler, uses feature_importances)
model, feature_columns, auc, feature_importances = train_model(df)

# ---------- Overview ----------
if page == "📊 Overview":
    st.title("Churn Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Overall churn rate", f"{df['churned'].mean()*100:.1f}%")
    c3.metric("Churned customers", f"{df['churned'].sum():,}")
    c4.metric("Model AUC (XGBoost)", f"{auc:.3f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        rate = df.groupby('subscription_type')['churned'].mean().sort_values(ascending=False) * 100
        fig = px.bar(rate, title="Churn Rate by Subscription Tier", labels={'value': 'Churn %', 'subscription_type': ''})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        rate = df.groupby('primary_device')['churned'].mean().sort_values(ascending=False) * 100
        fig = px.bar(rate, title="Churn Rate by Device", labels={'value': 'Churn %', 'primary_device': ''})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df, x='days_since_last_login', color='churned', barmode='overlay',
                            title="Days Since Last Login vs Churn", nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        rate = df.groupby('recommendation_source')['churned'].mean().sort_values(ascending=False) * 100
        fig = px.bar(rate, title="Churn Rate by Recommendation Source", labels={'value': 'Churn %', 'recommendation_source': ''})
        st.plotly_chart(fig, use_container_width=True)

# ---------- Driver Analysis ----------
elif page == "🔍 Driver Analysis":
    st.title("What Drives Churn (XGBoost Feature Importance)")
    st.caption("XGBoost feature importances — higher value = greater relative impact on the model's predictions.")

    top_n = st.slider("Number of features to show", 5, 25, 10)
    top_importances = feature_importances.head(top_n).sort_values()

    fig = px.bar(
        top_importances, orientation='h',
        title="Top Feature Importances for Churn Prediction",
        labels={'value': 'Importance Score', 'index': ''}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore a factor")
    factor = st.selectbox("Categorical factor", CAT_COLS)
    rate = df.groupby(factor)['churned'].mean().sort_values(ascending=False) * 100
    fig2 = px.bar(rate, labels={'value': 'Churn %', factor: ''}, title=f"Churn Rate by {factor}")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Batch Scoring ----------
elif page == "📥 Batch Scoring":
    st.title("Batch Risk Scoring")
    st.write("Every customer in the dataset, scored for churn risk via XGBoost and ranked from highest to lowest.")

    batch_df = df.copy()
    X_batch = pd.get_dummies(batch_df[NUM_COLS + CAT_COLS], columns=CAT_COLS)
    X_batch = X_batch.reindex(columns=feature_columns, fill_value=0)
    
    # XGBoost prediction directly on dataframe (no scaler needed)
    batch_df['churn_risk'] = model.predict_proba(X_batch)[:, 1]
    batch_df['risk_tier'] = pd.cut(
        batch_df['churn_risk'], bins=[0, 0.33, 0.66, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    tier_filter = st.multiselect(
        "Filter by risk tier", options=['Low', 'Medium', 'High'],
        default=['Low', 'Medium', 'High']
    )
    result = batch_df[batch_df['risk_tier'].isin(tier_filter)].sort_values('churn_risk', ascending=False)

    st.dataframe(result, use_container_width=True)

    st.download_button(
        "Download scored results",
        result.to_csv(index=False).encode('utf-8'),
        "scored_customers_xgb.csv",
        "text/csv"
    )

    risk_counts = batch_df['risk_tier'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Tier Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ---------- What-If Simulator ----------
elif page == "🎛️ What-If Simulator":
    st.title("Single Customer What-If Simulator")
    st.write("Adjust a customer's profile and watch predicted XGBoost churn risk update in real time.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 15, 75, 35)
        account_age_months = st.slider("Account age (months)", 0, 90, 24)
        session_count = st.slider("Session count", 0, 30, 3)
        avg_watch_time = st.slider("Avg watch time (min/week)", 0, 700, 200)
        watch_sessions = st.slider("Watch sessions/week", 0, 25, 5)
    with col2:
        completion_rate = st.slider("Completion rate (%)", 0, 100, 70)
        avg_rating_given = st.slider("Avg rating given", 1, 5, 4)
        app_rating = st.slider("App rating", 1, 5, 4)
        rec_click_rate = st.slider("Recommendation click rate (%)", 0, 100, 30)
        days_since_login = st.slider("Days since last login", 0, 90, 5)
    with col3:
        gender = st.selectbox("Gender", sorted(df['gender'].unique()))
        region = st.selectbox("Region", sorted(df['region'].unique()))
        subscription_type = st.selectbox("Subscription tier", sorted(df['subscription_type'].unique()))
        payment_method = st.selectbox("Payment method", sorted(df['payment_method'].unique()))
        primary_device = st.selectbox("Primary device", sorted(df['primary_device'].unique()))
        favorite_genre = st.selectbox("Favorite genre", sorted(df['favorite_genre'].unique()))
        time_of_day = st.selectbox("Time of day", sorted(df['time_of_day'].unique()))
        recommendation_source = st.selectbox("Recommendation source", sorted(df['recommendation_source'].unique()))

    input_dict = {
        'age': age, 'account_age_months': account_age_months, 'session_count': session_count,
        'avg_watch_time_minutes_per_week': avg_watch_time, 'watch_sessions_per_week': watch_sessions,
        'completion_rate': completion_rate, 'avg_rating_given': avg_rating_given,
        'app_rating': app_rating, 'recommendation_click_rate': rec_click_rate,
        'days_since_last_login': days_since_login,
        'gender': gender, 'region': region, 'subscription_type': subscription_type,
        'payment_method': payment_method, 'primary_device': primary_device,
        'favorite_genre': favorite_genre, 'time_of_day': time_of_day,
        'recommendation_source': recommendation_source
    }

    row = build_feature_row(input_dict, feature_columns)
    
    # Predict directly without feature scaling
    risk = model.predict_proba(row)[0, 1]

    st.divider()
    st.subheader("Predicted Churn Risk")
    st.progress(min(int(risk * 100), 100))
    if risk < 0.33:
        st.success(f"{risk*100:.1f}% — Low risk")
    elif risk < 0.66:
        st.warning(f"{risk*100:.1f}% — Medium risk")
    else:
        st.error(f"{risk*100:.1f}% — High risk")
