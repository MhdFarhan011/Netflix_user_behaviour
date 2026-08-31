import streamlit as st
import pandas as pd
import plotly.express as px
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Netflix Churn Insights",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# NETFLIX COLORS
# =========================================================

NETFLIX_RED = "#E50914"
BLACK = "#141414"
DARK_GRAY = "#1F1F1F"
LIGHT_GRAY = "#B3B3B3"
WHITE = "#FFFFFF"
GREEN = "#46D369"
YELLOW = "#F5C518"


# =========================================================
# FEATURES
# =========================================================

NUM_COLS = [
    "age",
    "account_age_months",
    "session_count",
    "avg_watch_time_minutes_per_week",
    "watch_sessions_per_week",
    "completion_rate",
    "avg_rating_given",
    "app_rating",
    "recommendation_click_rate",
    "days_since_last_login"
]

CAT_COLS = [
    "gender",
    "region",
    "subscription_type",
    "payment_method",
    "primary_device",
    "favorite_genre",
    "time_of_day",
    "recommendation_source"
]


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "netflix_user_behavior_churn_50000.csv"
    )


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "netflix_user_behaviour_model.pkl"
    )

    scaler = joblib.load(
        "SCALER2.pkl"
    )

    return model, scaler


# =========================================================
# LOAD MODEL + DATA
# =========================================================

df = load_data()

model, scaler = load_model()


# =========================================================
# GET MODEL FEATURE COLUMNS
# =========================================================

# XGBoost normally stores the feature names used during
# training when the model was trained with a DataFrame.

if hasattr(model, "feature_names_in_"):

    FEATURE_COLUMNS = list(
        model.feature_names_in_
    )

elif hasattr(model, "get_booster"):

    FEATURE_COLUMNS = (
        model.get_booster()
        .feature_names
    )

else:

    FEATURE_COLUMNS = None


# =========================================================
# PREPROCESS DATA
# =========================================================

def preprocess_data(data):

    X = data[
        NUM_COLS + CAT_COLS
    ].copy()

    X = pd.get_dummies(
        X,
        columns=CAT_COLS,
        drop_first=True
    )

    # Make sure columns are in exactly the
    # same order as during model training.

    if FEATURE_COLUMNS is not None:

        X = X.reindex(
            columns=FEATURE_COLUMNS,
            fill_value=0
        )

    # Apply the same scaler used during training.

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    return X_scaled


# =========================================================
# CREATE CUSTOMER ROW
# =========================================================

def build_feature_row(
    input_dict
):

    row = pd.DataFrame(
        [input_dict]
    )

    row_encoded = pd.get_dummies(
        row,
        columns=CAT_COLS,
        drop_first=True
    )

    # Match training feature columns exactly.

    if FEATURE_COLUMNS is not None:

        row_encoded = row_encoded.reindex(
            columns=FEATURE_COLUMNS,
            fill_value=0
        )

    # Apply saved scaler.

    row_scaled = scaler.transform(
        row_encoded
    )

    row_scaled = pd.DataFrame(
        row_scaled,
        columns=row_encoded.columns
    )

    return row_scaled


# =========================================================
# PLOTLY NETFLIX THEME
# =========================================================

def netflix_chart(fig):

    fig.update_layout(
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(
            color=WHITE
        ),
        title_font=dict(
            color=WHITE,
            size=20
        ),
        xaxis=dict(
            gridcolor="#333333",
            color=WHITE
        ),
        yaxis=dict(
            gridcolor="#333333",
            color=WHITE
        ),
        legend=dict(
            font=dict(
                color=WHITE
            )
        )
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 NETFLIX")

st.sidebar.caption(
    "CHURN INSIGHTS"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "🔍 Driver Analysis",
        "📥 Batch Scoring",
        "🎛️ What-If Simulator"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    "Powered by XGBoost"
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🎬 NETFLIX")

st.subheader(
    "User Behavior & Churn Intelligence"
)

st.caption(
    "Analyze customer behavior, identify churn drivers "
    "and predict individual customer churn risk."
)

st.divider()


# =========================================================
# OVERVIEW
# =========================================================

if page == "📊 Overview":

    st.header("📊 Churn Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Customers",
        f"{len(df):,}"
    )

    c2.metric(
        "Overall Churn Rate",
        f"{df['churned'].mean() * 100:.1f}%"
    )

    c3.metric(
        "Churned Customers",
        f"{df['churned'].sum():,}"
    )

    st.divider()

    # -----------------------------------------
    # Subscription
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        rate = (
            df.groupby("subscription_type")["churned"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig = px.bar(
            rate,
            title="Churn Rate by Subscription Tier",
            labels={
                "value": "Churn %",
                "subscription_type": "Subscription"
            }
        )

        fig.update_traces(
            marker_color=NETFLIX_RED
        )

        st.plotly_chart(
            netflix_chart(fig),
            use_container_width=True
        )

    # -----------------------------------------
    # Device
    # -----------------------------------------

    with col2:

        rate = (
            df.groupby("primary_device")["churned"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig = px.bar(
            rate,
            title="Churn Rate by Device",
            labels={
                "value": "Churn %",
                "primary_device": "Device"
            }
        )

        fig.update_traces(
            marker_color=NETFLIX_RED
        )

        st.plotly_chart(
            netflix_chart(fig),
            use_container_width=True
        )

    # -----------------------------------------
    # Login Activity
    # -----------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            df,
            x="days_since_last_login",
            color="churned",
            barmode="overlay",
            nbins=40,
            title="Days Since Last Login vs Churn"
        )

        st.plotly_chart(
            netflix_chart(fig),
            use_container_width=True
        )

    # -----------------------------------------
    # Recommendation Source
    # -----------------------------------------

    with col4:

        rate = (
            df.groupby("recommendation_source")["churned"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig = px.bar(
            rate,
            title="Churn Rate by Recommendation Source",
            labels={
                "value": "Churn %",
                "recommendation_source": "Source"
            }
        )

        fig.update_traces(
            marker_color=NETFLIX_RED
        )

        st.plotly_chart(
            netflix_chart(fig),
            use_container_width=True
        )


# =========================================================
# DRIVER ANALYSIS
# =========================================================

elif page == "🔍 Driver Analysis":

    st.header("🔍 What Drives Churn")

    st.caption(
        "The saved XGBoost model's feature importance "
        "shows which features contribute most to predictions."
    )

    # -----------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_values = (
            model.feature_importances_
        )

        importance = pd.Series(
            importance_values,
            index=FEATURE_COLUMNS
        ).sort_values(
            ascending=False
        )

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=10,
            value=8
        )

        top_importances = (
            importance
            .head(top_n)
            .sort_values()
        )

        fig = px.bar(
            top_importances,
            orientation="h",
            title="Top Churn Drivers",
            labels={
                "value": "Importance",
                "index": "Feature"
            }
        )

        fig.update_traces(
            marker_color=NETFLIX_RED
        )

        st.plotly_chart(
            netflix_chart(fig),
            use_container_width=True
        )

    st.divider()

    # -----------------------------------------
    # CATEGORICAL FACTOR ANALYSIS
    # -----------------------------------------

    st.subheader(
        "Explore Customer Factors"
    )

    factor = st.selectbox(
        "Select a factor",
        CAT_COLS
    )

    rate = (
        df.groupby(factor)["churned"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    fig2 = px.bar(
        rate,
        title=f"Churn Rate by {factor}",
        labels={
            "value": "Churn %",
            factor: factor
        }
    )

    fig2.update_traces(
        marker_color=NETFLIX_RED
    )

    st.plotly_chart(
        netflix_chart(fig2),
        use_container_width=True
    )


# =========================================================
# BATCH SCORING
# =========================================================

elif page == "📥 Batch Scoring":

    st.header("📥 Batch Risk Scoring")

    st.write(
        "Every customer is scored using the saved "
        "XGBoost model and ranked by predicted churn probability."
    )

    # -----------------------------------------
    # PREPROCESS DATA
    # -----------------------------------------

    X_batch = preprocess_data(
        df
    )

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    batch_df = df.copy()

    batch_df["churn_risk"] = (
        model.predict_proba(
            X_batch
        )[:, 1]
    )

    # -----------------------------------------
    # RISK TIER
    # -----------------------------------------

    batch_df["risk_tier"] = pd.cut(
        batch_df["churn_risk"],
        bins=[
            0,
            0.33,
            0.66,
            1
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    tier_filter = st.multiselect(
        "Risk Tier",
        [
            "Low",
            "Medium",
            "High"
        ],
        default=[
            "Low",
            "Medium",
            "High"
        ]
    )

    result = (
        batch_df[
            batch_df["risk_tier"].isin(
                tier_filter
            )
        ]
        .sort_values(
            "churn_risk",
            ascending=False
        )
    )

    st.dataframe(
        result,
        use_container_width=True
    )

    # -----------------------------------------
    # DOWNLOAD
    # -----------------------------------------

    st.download_button(
        label="⬇️ Download Scored Customers",
        data=result.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="scored_customers_xgb.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------------------
    # RISK DISTRIBUTION
    # -----------------------------------------

    risk_counts = (
        batch_df["risk_tier"]
        .value_counts()
    )

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Tier Distribution"
    )

    fig.update_traces(
        marker=dict(
            colors=[
                GREEN,
                YELLOW,
                NETFLIX_RED
            ]
        )
    )

    st.plotly_chart(
        netflix_chart(fig),
        use_container_width=True
    )


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "🎛️ What-If Simulator":

    st.header(
        "🎛️ Single Customer What-If Simulator"
    )

    st.write(
        "Change the customer's profile to see "
        "how the predicted churn probability changes."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------
    # CUSTOMER ACTIVITY
    # -----------------------------------------

    with col1:

        st.subheader("Customer Activity")

        age = st.slider(
            "Age",
            15,
            75,
            35
        )

        account_age_months = st.slider(
            "Account Age (months)",
            0,
            90,
            24
        )

        session_count = st.slider(
            "Session Count",
            0,
            30,
            3
        )

        avg_watch_time = st.slider(
            "Avg Watch Time (min/week)",
            0,
            700,
            200
        )

        watch_sessions = st.slider(
            "Watch Sessions / Week",
            0,
            25,
            5
        )

    # -----------------------------------------
    # ENGAGEMENT
    # -----------------------------------------

    with col2:

        st.subheader("Engagement")

        completion_rate = st.slider(
            "Completion Rate (%)",
            0,
            100,
            70
        )

        avg_rating_given = st.slider(
            "Average Rating Given",
            1,
            5,
            4
        )

        app_rating = st.slider(
            "App Rating",
            1,
            5,
            4
        )

        rec_click_rate = st.slider(
            "Recommendation Click Rate (%)",
            0,
            100,
            30
        )

        days_since_login = st.slider(
            "Days Since Last Login",
            0,
            90,
            5
        )

    # -----------------------------------------
    # PROFILE
    # -----------------------------------------

    with col3:

        st.subheader("Profile")

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["region"].unique())
        )

        subscription_type = st.selectbox(
            "Subscription Type",
            sorted(df["subscription_type"].unique())
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(df["payment_method"].unique())
        )

        primary_device = st.selectbox(
            "Primary Device",
            sorted(df["primary_device"].unique())
        )

        favorite_genre = st.selectbox(
            "Favorite Genre",
            sorted(df["favorite_genre"].unique())
        )

        time_of_day = st.selectbox(
            "Time of Day",
            sorted(df["time_of_day"].unique())
        )

        recommendation_source = st.selectbox(
            "Recommendation Source",
            sorted(df["recommendation_source"].unique())
        )

    # -----------------------------------------
    # INPUT DICTIONARY
    # -----------------------------------------

    input_dict = {

        "age": age,

        "account_age_months":
            account_age_months,

        "session_count":
            session_count,

        "avg_watch_time_minutes_per_week":
            avg_watch_time,

        "watch_sessions_per_week":
            watch_sessions,

        "completion_rate":
            completion_rate,

        "avg_rating_given":
            avg_rating_given,

        "app_rating":
            app_rating,

        "recommendation_click_rate":
            rec_click_rate,

        "days_since_last_login":
            days_since_login,

        "gender":
            gender,

        "region":
            region,

        "subscription_type":
            subscription_type,

        "payment_method":
            payment_method,

        "primary_device":
            primary_device,

        "favorite_genre":
            favorite_genre,

        "time_of_day":
            time_of_day,

        "recommendation_source":
            recommendation_source
    }

    # -----------------------------------------
    # BUILD MODEL INPUT
    # -----------------------------------------

    row = build_feature_row(
        input_dict
    )

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    risk = model.predict_proba(
        row
    )[0, 1]

    st.divider()

    st.subheader(
        "🎬 Predicted Churn Risk"
    )

    st.progress(
        int(risk * 100)
    )

    # -----------------------------------------
    # RISK DISPLAY
    # -----------------------------------------

    if risk < 0.33:

        st.success(
            f"🟢 LOW RISK — {risk * 100:.1f}%"
        )

        st.write(
            "This customer has a relatively low "
            "probability of churning."
        )

    elif risk < 0.66:

        st.warning(
            f"🟡 MEDIUM RISK — {risk * 100:.1f}%"
        )

        st.write(
            "This customer shows moderate "
            "churn risk and may need attention."
        )

    else:

        st.error(
            f"🔴 HIGH RISK — {risk * 100:.1f}%"
        )

        st.write(
            "This customer has a high predicted "
            "probability of churning."
        )
