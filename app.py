import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FILES

DATASET_FILE = "Employee_Salary_Dataset (1).csv"

LINEAR_MODEL_FILE = "linear_model (2).pkl"
POLY_MODEL_FILE = "polynomial_model (2).pkl"
POLY_FEATURE_FILE = "polynomial_features (2).pkl"

# FEATURES

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]

# LOAD DATASET

try:

    df = pd.read_csv(DATASET_FILE)

except Exception as e:

    st.error(
        f"Unable to load {DATASET_FILE}. "
        f"Please make sure the CSV file is in the same folder as app.py."
    )

    st.stop()

# CLEAN DATA

df.columns = df.columns.str.strip()

for column in df.columns:

    if df[column].dtype == "object":

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )


# FIND SALARY COLUMN

salary_column = None

possible_salary_columns = [
    "Salary",
    "salary",
    "Salary_USD",
    "Monthly Salary",
    "Annual Salary"
]

for column in possible_salary_columns:

    if column in df.columns:

        salary_column = column
        break


if salary_column is None:

    st.error(
        "Salary column was not found in your dataset. "
        "Please make sure your CSV contains a column named 'Salary'."
    )

    st.stop()

# CHECK REQUIRED COLUMNS

missing_columns = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns in dataset: "
        + ", ".join(missing_columns)
    )

    st.stop()

# ENCODING

gender_values = sorted(
    df["Gender"]
    .dropna()
    .astype(str)
    .unique()
)

education_values = sorted(
    df["Education Level"]
    .dropna()
    .astype(str)
    .unique()
)

job_values = sorted(
    df["Job Title"]
    .dropna()
    .astype(str)
    .unique()
)

# Gender Mapping

gender_map = {}

for index, value in enumerate(gender_values):

    gender_map[value] = index

# Education Mapping

education_map = {}

for index, value in enumerate(education_values):

    education_map[value] = index

# Job Mapping

job_map = {}

for index, value in enumerate(job_values):

    job_map[value] = index

# CREATE NUMERIC DATASET

model_df = df.copy()

model_df["Gender"] = model_df["Gender"].map(
    gender_map
)

model_df["Education Level"] = model_df[
    "Education Level"
].map(
    education_map
)

model_df["Job Title"] = model_df[
    "Job Title"
].map(
    job_map
)


model_df[salary_column] = pd.to_numeric(
    model_df[salary_column],
    errors="coerce"
)


model_df = model_df.dropna(
    subset=FEATURE_COLUMNS + [salary_column]
).copy()

# LOAD ORIGINAL MODELS

try:

    linear = joblib.load(
        LINEAR_MODEL_FILE
    )

    poly_model = joblib.load(
        POLY_MODEL_FILE
    )

    poly = joblib.load(
        POLY_FEATURE_FILE
    )

except Exception as e:

    st.error(
        "Unable to load Linear / Polynomial model files. "
        "Make sure the .pkl files are in the same folder as app.py."
    )

    st.stop()


# TRAIN SAMPLE DECISION TREE + RANDOM FOREST

X = model_df[FEATURE_COLUMNS]

y = model_df[salary_column]


if len(model_df) >= 10:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    decision_tree = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )

    random_forest = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )

    decision_tree.fit(
        X_train,
        y_train
    )

    random_forest.fit(
        X_train,
        y_train
    )

else:

    decision_tree = None
    random_forest = None

    X_train = None
    X_test = None
    y_train = None
    y_test = None

# CSS DESIGN

st.markdown(
    """
<style>

/* ================================
   MAIN APP
================================ */

.stApp {
    background:
        linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef4ff 50%,
            #f8f5ff 100%
        );
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #111827 0%,
            #172554 45%,
            #312e81 100%
        );
}


[data-testid="stSidebar"] * {
    color: white !important;
}


[data-testid="stSidebar"] hr {

    border-color:
        rgba(255,255,255,0.25);

}


.sidebar-logo {

    font-size: 55px;
    text-align: center;
    margin-bottom: 5px;

}


.sidebar-title {

    color: white !important;
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 1px;

}


.sidebar-subtitle {

    color: #c7d2fe !important;
    text-align: center;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 20px;

}


.sidebar-section {

    color: #c7d2fe !important;
    font-size: 14px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;

}


.sidebar-item {

    color: white !important;
    font-size: 14px;
    margin-bottom: 7px;

}


/* ================================
   TITLES
================================ */

.main-title {

    font-size: 44px;
    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #111827,
            #4338ca,
            #7c3aed
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 2px;

}


.subtitle {

    color: #64748b;
    font-size: 18px;
    margin-bottom: 28px;

}


.section-title {

    color: #1e1b4b;
    font-size: 25px;
    font-weight: 850;
    margin-top: 20px;
    margin-bottom: 18px;

}


/* ================================
   METRIC CARDS
================================ */

[data-testid="stMetric"] {

    background:
        rgba(255,255,255,0.85);

    border-radius: 18px;

    padding: 18px;

    border:
        1px solid rgba(99,102,241,0.15);

    box-shadow:
        0 8px 25px rgba(30,41,59,0.08);

}


/* ================================
   BUTTON
================================ */

.stButton > button {

    width: 100%;

    border: none;

    border-radius: 14px;

    padding: 14px;

    font-size: 16px;

    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );

    color: white !important;

    box-shadow:
        0 7px 20px rgba(79,70,229,0.30);

}


.stButton > button:hover {

    background:
        linear-gradient(
            90deg,
            #4338ca,
            #6d28d9
        );

    color: white !important;

}


/* ================================
   INPUT LABELS
================================ */

.stApp label {

    color: #1e293b !important;
    font-weight: 700 !important;

}


[data-testid="stWidgetLabel"] p {

    color: #1e293b !important;
    font-weight: 700 !important;

}


/* ================================
   PREDICTION CARD
================================ */

.prediction-card {

    padding: 32px;

    border-radius: 25px;

    text-align: center;

    margin-top: 25px;
    margin-bottom: 25px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff,
            #ede9fe
        );

    border:
        2px solid #6366f1;

    box-shadow:
        0 15px 40px rgba(79,70,229,0.15);

}


.prediction-title {

    font-size: 21px;
    font-weight: 900;
    color: #4338ca;

}


.prediction-label {

    font-size: 17px;
    font-weight: 750;
    margin-top: 18px;
    color: #334155;

}


.prediction-value {

    font-size: 52px;
    font-weight: 950;
    color: #4f46e5;
    margin: 8px;

}


.prediction-unit {

    font-size: 15px;
    color: #64748b;

}


.prediction-info {

    font-size: 16px;
    font-weight: 750;
    color: #334155;

}


/* ================================
   INFO CARDS
================================ */

.info-card {

    background:
        rgba(255,255,255,0.9);

    padding: 22px;

    border-radius: 18px;

    border-left:
        5px solid #6366f1;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.07);

    margin-bottom: 15px;

}


.info-title {

    font-size: 18px;
    font-weight: 850;
    color: #312e81;

}


.info-text {

    font-size: 14px;
    color: #64748b;
    margin-top: 5px;

}


/* ================================
   FOOTER
================================ */

.footer {

    text-align: center;

    color: #64748b;

    padding: 35px;

    font-size: 14px;

}

</style>
""",
    unsafe_allow_html=True
)

# SIDEBAR

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-logo">
💼
</div>

<div class="sidebar-title">
EMPLOYEE SALARY
</div>

<div class="sidebar-title">
PREDICTION
</div>

<div class="sidebar-subtitle">
Machine Learning Analytics Dashboard
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📊 Analytics",
            "🤖 Models",
            "📈 Forecast"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
<div class="sidebar-section">
🤖 MACHINE LEARNING MODELS
</div>

<div class="sidebar-item">
📘 Linear Regression
</div>

<div class="sidebar-item">
📗 Polynomial Regression
</div>

<div class="sidebar-item">
🌳 Decision Tree
</div>

<div class="sidebar-item">
🌲 Random Forest
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
<div class="sidebar-section">
📌 PROJECT FEATURES
</div>

<div class="sidebar-item">
✓ Salary Prediction
</div>

<div class="sidebar-item">
✓ Data Analytics
</div>

<div class="sidebar-item">
✓ Model Comparison
</div>

<div class="sidebar-item">
✓ Salary Forecast
</div>
""",
        unsafe_allow_html=True
    )

# DASHBOARD

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">💼 Employee Salary Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predict employee salary using Machine Learning</div>',
        unsafe_allow_html=True
    )

    # TOP KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Employees",
        f"{len(df):,}"
    )

    c2.metric(
        "💰 Average Salary",
        f"₹ {df[salary_column].mean():,.0f}"
    )

    c3.metric(
        "📈 Maximum Salary",
        f"₹ {df[salary_column].max():,.0f}"
    )

    c4.metric(
        "🎓 Education Levels",
        df["Education Level"].nunique()
    )


    st.markdown(
        '<div class="section-title">👤 Enter Employee Details</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=70,
            value=30,
            step=1
        )

        gender = st.selectbox(
            "👤 Gender",
            gender_values
        )

        education = st.selectbox(
            "🎓 Education Level",
            education_values
        )


    with col2:

        job = st.selectbox(
            "💼 Job Title",
            job_values
        )

        experience = st.number_input(
            "⏳ Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )


    model_choice = st.selectbox(
        "🤖 Select Machine Learning Model",
        [
            "Random Forest",
            "Decision Tree",
            "Linear Regression",
            "Polynomial Regression"
        ]
    )


    predict = st.button(
        "💰 PREDICT EMPLOYEE SALARY"
    )


    if predict:

        gender_encoded = gender_map[gender]

        education_encoded = education_map[
            education
        ]

        job_encoded = job_map[job]


        input_df = pd.DataFrame(
            [[
                age,
                gender_encoded,
                education_encoded,
                job_encoded,
                experience
            ]],
            columns=FEATURE_COLUMNS
        )


        if model_choice == "Linear Regression":

            prediction = linear.predict(
                input_df
            )[0]


        elif model_choice == "Polynomial Regression":

            prediction = poly_model.predict(
                poly.transform(input_df)
            )[0]


        elif model_choice == "Decision Tree":

            if decision_tree is None:

                st.error(
                    "Decision Tree requires enough dataset records."
                )

                st.stop()

            prediction = decision_tree.predict(
                input_df
            )[0]


        else:

            if random_forest is None:

                st.error(
                    "Random Forest requires enough dataset records."
                )

                st.stop()

            prediction = random_forest.predict(
                input_df
            )[0]


        prediction = max(
            0.0,
            float(prediction)
        )

        # SALARY LEVEL

        q1 = df[salary_column].quantile(0.33)

        q2 = df[salary_column].quantile(0.66)


        if prediction < q1:

            salary_level = "LOW"
            salary_icon = "🔵"

        elif prediction < q2:

            salary_level = "MEDIUM"
            salary_icon = "🟢"

        else:

            salary_level = "HIGH"
            salary_icon = "🟣"

        # PREDICTION CARD

        prediction_html = f"""
<div class="prediction-card">

<div class="prediction-title">
💼 SALARY PREDICTION RESULT
</div>

<div class="prediction-label">
PREDICTED SALARY
</div>

<div class="prediction-value">
₹ {prediction:,.2f}
</div>

<div class="prediction-unit">
Estimated Employee Salary
</div>

<br>

<div class="prediction-info">
{salary_icon} Salary Level: {salary_level}
</div>

<br>

<div class="prediction-info">
🤖 Model Used: {model_choice}
</div>

</div>
"""


        st.markdown(
            prediction_html,
            unsafe_allow_html=True
        )

        # INPUT SUMMARY

        st.markdown(
            '<div class="section-title">📋 Employee Information</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "🎂 Age",
            age
        )

        c2.metric(
            "👤 Gender",
            gender
        )

        c3.metric(
            "🎓 Education",
            education
        )

        c4.metric(
            "💼 Job",
            job
        )


        c5, c6 = st.columns(2)


        c5.metric(
            "⏳ Experience",
            f"{experience:.1f} Years"
        )

        c6.metric(
            "💰 Predicted Salary",
            f"₹ {prediction:,.0f}"
        )

        # VISUALIZATION

        st.markdown(
            '<div class="section-title">📊 Salary Visualization</div>',
            unsafe_allow_html=True
        )


        fig, ax = plt.subplots(
            figsize=(10, 4)
        )


        ax.bar(
            ["Predicted Salary"],
            [prediction]
        )


        ax.set_ylabel(
            "Salary"
        )

        ax.set_title(
            f"Predicted Salary for {job}"
        )


        st.pyplot(fig)

        plt.close(fig)

        # FEATURE IMPORTANCE

        if random_forest is not None:

            st.markdown(
                '<div class="section-title">🔍 Important Salary Factors</div>',
                unsafe_allow_html=True
            )


            importance = pd.DataFrame({

                "Feature": FEATURE_COLUMNS,

                "Importance":
                    random_forest.feature_importances_

            }).sort_values(
                "Importance",
                ascending=False
            )


            fig, ax = plt.subplots(
                figsize=(10, 4)
            )


            ax.barh(
                importance["Feature"],
                importance["Importance"]
            )


            ax.set_xlabel(
                "Importance"
            )

            ax.set_title(
                "Random Forest Feature Importance"
            )

            ax.invert_yaxis()


            st.pyplot(fig)

            plt.close(fig)

# ANALYTICS

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">📊 Salary Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore employee salary patterns and trends</div>',
        unsafe_allow_html=True
    )

    # KPI

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "👥 Employees",
        f"{len(df):,}"
    )

    c2.metric(
        "💰 Average Salary",
        f"₹ {df[salary_column].mean():,.0f}"
    )

    c3.metric(
        "📉 Minimum Salary",
        f"₹ {df[salary_column].min():,.0f}"
    )

    c4.metric(
        "📈 Maximum Salary",
        f"₹ {df[salary_column].max():,.0f}"
    )


    analysis = st.selectbox(
        "📊 Choose Analysis",
        [
            "Salary Distribution",
            "Age vs Salary",
            "Experience vs Salary",
            "Education vs Salary",
            "Job-wise Average Salary",
            "Gender-wise Average Salary"
        ]
    )

    # SALARY DISTRIBUTION

    if analysis == "Salary Distribution":

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.hist(
            df[salary_column],
            bins=30
        )


        ax.set_xlabel(
            "Salary"
        )

        ax.set_ylabel(
            "Number of Employees"
        )

        ax.set_title(
            "Employee Salary Distribution"
        )


        st.pyplot(fig)

        plt.close(fig)

    # AGE VS SALARY

    elif analysis == "Age vs Salary":

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.scatter(
            df["Age"],
            df[salary_column],
            alpha=0.5
        )


        ax.set_xlabel(
            "Age"
        )

        ax.set_ylabel(
            "Salary"
        )

        ax.set_title(
            "Age vs Employee Salary"
        )


        st.pyplot(fig)

        plt.close(fig)

    # EXPERIENCE VS SALARY

    elif analysis == "Experience vs Salary":

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.scatter(
            df["Years of Experience"],
            df[salary_column],
            alpha=0.5
        )


        ax.set_xlabel(
            "Years of Experience"
        )

        ax.set_ylabel(
            "Salary"
        )

        ax.set_title(
            "Experience vs Employee Salary"
        )


        st.pyplot(fig)

        plt.close(fig)

    # EDUCATION VS SALARY

    elif analysis == "Education vs Salary":

        education_salary = (
            df.groupby(
                "Education Level"
            )[salary_column]
            .mean()
            .sort_values(
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.bar(
            education_salary.index,
            education_salary.values
        )


        ax.set_xlabel(
            "Education Level"
        )

        ax.set_ylabel(
            "Average Salary"
        )

        ax.set_title(
            "Education Level vs Average Salary"
        )


        st.pyplot(fig)

        plt.close(fig)

    # JOB-WISE SALARY

    elif analysis == "Job-wise Average Salary":

        job_salary = (
            df.groupby(
                "Job Title"
            )[salary_column]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(15)
        )


        fig, ax = plt.subplots(
            figsize=(11, 6)
        )


        ax.bar(
            job_salary.index.astype(str),
            job_salary.values
        )


        ax.set_xlabel(
            "Job Title"
        )

        ax.set_ylabel(
            "Average Salary"
        )

        ax.set_title(
            "Top Job Titles by Average Salary"
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        st.pyplot(fig)

        plt.close(fig)

    # GENDER SALARY

    else:

        gender_salary = (
            df.groupby(
                "Gender"
            )[salary_column]
            .mean()
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        ax.bar(
            gender_salary.index,
            gender_salary.values
        )


        ax.set_xlabel(
            "Gender"
        )

        ax.set_ylabel(
            "Average Salary"
        )

        ax.set_title(
            "Gender-wise Average Salary"
        )


        st.pyplot(fig)

        plt.close(fig)

# MODELS

elif page == "🤖 Models":

    st.markdown(
        '<div class="main-title">🤖 Machine Learning Models</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Compare different salary prediction algorithms</div>',
        unsafe_allow_html=True
    )


    if len(model_df) < 10:

        st.error(
            "Not enough records for model comparison."
        )

        st.stop()

    # TRAIN TEST

    X = model_df[
        FEATURE_COLUMNS
    ]

    y = model_df[
        salary_column
    ]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # PREDICTIONS

    linear_prediction = linear.predict(
        X_test
    )


    polynomial_prediction = poly_model.predict(
        poly.transform(X_test)
    )


    tree_prediction = decision_tree.predict(
        X_test
    )


    forest_prediction = random_forest.predict(
        X_test
    )

    # COMPARISON

    comparison = pd.DataFrame({

        "Model": [

            "Linear Regression",

            "Polynomial Regression",

            "Decision Tree",

            "Random Forest"

        ],

        "R² Score": [

            r2_score(
                y_test,
                linear_prediction
            ),

            r2_score(
                y_test,
                polynomial_prediction
            ),

            r2_score(
                y_test,
                tree_prediction
            ),

            r2_score(
                y_test,
                forest_prediction
            )

        ],

        "MAE": [

            mean_absolute_error(
                y_test,
                linear_prediction
            ),

            mean_absolute_error(
                y_test,
                polynomial_prediction
            ),

            mean_absolute_error(
                y_test,
                tree_prediction
            ),

            mean_absolute_error(
                y_test,
                forest_prediction
            )

        ],

        "RMSE": [

            np.sqrt(
                mean_squared_error(
                    y_test,
                    linear_prediction
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test,
                    polynomial_prediction
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test,
                    tree_prediction
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test,
                    forest_prediction
                )
            )

        ]

    })


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # BEST MODEL

    best_model = comparison.loc[
        comparison["R² Score"].idxmax()
    ]


    st.success(
        f"🏆 Best Model: {best_model['Model']}  |  "
        f"R² Score: {best_model['R² Score']:.4f}"
    )

    # MODEL GRAPH

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.bar(
        comparison["Model"],
        comparison["R² Score"]
    )


    ax.set_ylabel(
        "R² Score"
    )

    ax.set_title(
        "Salary Prediction Model Performance"
    )


    plt.xticks(
        rotation=20
    )


    st.pyplot(fig)

    plt.close(fig)

    # FEATURE IMPORTANCE

    st.markdown(
        '<div class="section-title">🔍 Salary Feature Importance</div>',
        unsafe_allow_html=True
    )


    importance = pd.DataFrame({

        "Feature":
            FEATURE_COLUMNS,

        "Importance":
            random_forest.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )


    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.invert_yaxis()


    st.pyplot(fig)

    plt.close(fig)

# FORECAST

elif page == "📈 Forecast":

    st.markdown(
        '<div class="main-title">📈 Salary Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore how salary may change with increasing experience</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        forecast_age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=70,
            value=30,
            step=1,
            key="forecast_age"
        )


        forecast_gender = st.selectbox(
            "👤 Gender",
            gender_values,
            key="forecast_gender"
        )


        forecast_education = st.selectbox(
            "🎓 Education Level",
            education_values,
            key="forecast_education"
        )


    with col2:

        forecast_job = st.selectbox(
            "💼 Job Title",
            job_values,
            key="forecast_job"
        )


        forecast_start = st.number_input(
            "⏳ Starting Experience",
            min_value=0.0,
            max_value=40.0,
            value=2.0,
            step=1.0
        )


        forecast_years = st.slider(
            "🔮 Forecast Experience Range",
            min_value=1,
            max_value=15,
            value=5
        )


    forecast_model = st.selectbox(
        "🤖 Forecast Model",
        [
            "Random Forest",
            "Linear Regression",
            "Polynomial Regression"
        ]
    )


    if st.button(
        "📈 GENERATE SALARY FORECAST"
    ):


        gender_encoded = gender_map[
            forecast_gender
        ]


        education_encoded = education_map[
            forecast_education
        ]


        job_encoded = job_map[
            forecast_job
        ]


        years = [

            forecast_start + i

            for i in range(
                int(forecast_years)
            )

        ]


        predictions = []


        for experience_value in years:


            future_input = pd.DataFrame(

                [[
                    forecast_age,
                    gender_encoded,
                    education_encoded,
                    job_encoded,
                    experience_value
                ]],

                columns=FEATURE_COLUMNS

            )


            if forecast_model == "Random Forest":

                predicted = random_forest.predict(
                    future_input
                )[0]


            elif forecast_model == "Linear Regression":

                predicted = linear.predict(
                    future_input
                )[0]


            else:

                predicted = poly_model.predict(
                    poly.transform(
                        future_input
                    )
                )[0]


            predictions.append(
                max(
                    0.0,
                    float(predicted)
                )
            )


        forecast_df = pd.DataFrame({

            "Experience (Years)": years,

            "Predicted Salary": predictions

        })


        st.markdown(
            '<div class="section-title">🔮 Future Salary Prediction</div>',
            unsafe_allow_html=True
        )


        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # FORECAST GRAPH

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )


        ax.plot(
            forecast_df["Experience (Years)"],
            forecast_df["Predicted Salary"],
            marker="o",
            linewidth=3
        )


        ax.set_xlabel(
            "Years of Experience"
        )

        ax.set_ylabel(
            "Predicted Salary"
        )

        ax.set_title(
            f"{forecast_job} - Salary Forecast"
        )


        ax.grid()


        st.pyplot(fig)

        plt.close(fig)

        # FORECAST KPIs

        first_salary = predictions[0]

        last_salary = predictions[-1]


        if first_salary != 0:

            growth = (

                (
                    last_salary -
                    first_salary
                )
                /
                first_salary

            ) * 100

        else:

            growth = 0


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "💰 Starting Salary",
            f"₹ {first_salary:,.0f}"
        )


        c2.metric(
            "📈 Final Salary",
            f"₹ {last_salary:,.0f}"
        )


        c3.metric(
            "🚀 Salary Change",
            f"{growth:.2f}%"
        )

# FOOTER

st.markdown(
    """
<div class="footer">

💼 Employee Salary Prediction Dashboard

<br><br>

Python • Pandas • NumPy • Matplotlib • Scikit-learn • Joblib • Streamlit

<br>

Machine Learning Based Salary Analytics & Forecasting

</div>
""",
    unsafe_allow_html=True
)