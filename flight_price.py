# -*- coding: utf-8 -*-
"""
Combined and Enhanced Flight Price Predictor App
-------------------------------------------------
This Streamlit application predicts flight prices based on user input,
incorporating robust data preprocessing, model training pipelines,
target variable standardization (similar to Orange workflows),
and insightful visualizations.

Structure:
1.  Import Libraries
2.  Page Configuration
3.  Custom CSS
4.  Data Loading & Target Standardization
5.  Feature Definition
6.  Sidebar Controls
7.  Helper Function (Prediction Summary)
8.  Cached Model Training Functions
9.  Main Application Logic (Pages - Fixed SyntaxError) # <-- Updated
10. Footer
"""

# --- 1. Import Necessary Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore') # Suppress minor warnings

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Custom CSS ---
st.markdown("""
<style>
    /* Add padding */
    .main .block-container { padding: 2rem 3rem; }
    /* Header colors */
    h1, h2, h3 { color: #0072B2; }
    /* Sidebar background */
    .stSidebar { background-color: #f0f2f6; }
    /* Summary table styling */
    .prediction-summary th { background-color: #e6f3ff; text-align: left; }
    .prediction-summary td { text-align: left; }
    /* Page section highlighting */
    .page-section {
        background-color: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Style for tabs (if used) */
    button[data-baseweb="tab"] {
        font-size: 1.1rem; font-weight: 500; background-color: #e9ecef;
        border-radius: 0.5rem 0.5rem 0 0 !important; margin-right: 5px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #f8f9fa !important; border-bottom: 2px solid #0072B2 !important;
        color: #0072B2;
    }
    /* Reduce vertical space in sidebar sections */
     .stSidebar .stRadio > div { padding-bottom: 0.5rem; }
     .stSidebar .stMarkdown { padding-top: 0.5rem; padding-bottom: 0.5rem; }
     .stSidebar h3 { margin-bottom: 0.5rem; }

</style>
""", unsafe_allow_html=True)

# --- 4. Data Loading ---
@st.cache_data
def load_data():
    """Loads data, cleans, standardizes price, returns df and scaler."""
    try:
        file_path = 'Clean_Dataset.csv'
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        if 'flight' in df.columns: df = df.drop(columns=['flight'], errors='ignore')
        target_scaler = StandardScaler()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['price'], inplace=True)
        df['price_standardized'] = target_scaler.fit_transform(df[['price']])
        return df, target_scaler
    except FileNotFoundError:
        st.error(f"**Error: 'Clean_Dataset.csv' not found!** Ensure it's in the same folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}"); st.stop()

df_original, target_scaler = load_data()

# --- 5. Feature Definition ---
def define_feature_types(df):
    """Identifies categorical and numerical features, excluding targets and duration."""
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in ['price', 'price_standardized', 'duration']: # Duration removed
        if col in numerical_columns: numerical_columns.remove(col)
    return categorical_columns, numerical_columns

categorical_columns_all, numerical_columns_all = define_feature_types(df_original)

# --- 6. Sidebar Controls ---
st.sidebar.title("⚙️ Controls & Settings")
st.sidebar.subheader("Filter Data by Class (for Overview & Analysis)")
class_filter = st.sidebar.radio(
    "Select Flight Class:",
    ['All Classes', 'Economy', 'Business'], index=0, key='class_filter', label_visibility="collapsed"
)
if class_filter == 'Economy':
    df_display = df_original[df_original['class'] == 'Economy'].copy()
    st.sidebar.caption("Showing **Economy** flights for Overview/Analysis.")
elif class_filter == 'Business':
    df_display = df_original[df_original['class'] == 'Business'].copy()
    st.sidebar.caption("Showing **Business** flights for Overview/Analysis.")
else:
    df_display = df_original.copy()
    st.sidebar.caption("Showing **All Classes** for Overview/Analysis.")
if df_display.empty:
    st.warning(f"No data for '{class_filter}'."); st.stop()

st.sidebar.subheader("App Navigation")
app_mode = st.sidebar.radio(
    "Go to:",
    ["Data Overview", "Price Analysis", "Model Performance", "Predict Price"],
    index=0, key='app_mode', label_visibility="collapsed"
)

st.sidebar.subheader("Group # 1 Details")
st.sidebar.markdown("""
<div style="font-size: 0.9em; line-height: 1.4;">
- Govind Wattamwar: pgdsai25033<br>
- Mehul Jevani: pgdsai25028<br>
- Saurabh Chhabra: pgdsai25032<br>
- Sarfaraj Pansare: pgdsai25029<br>
- Amit Fegade: pgdsai25030
</div>
""", unsafe_allow_html=True)

# --- 7. Helper Function for Prediction Summary ---
def display_prediction_summary(input_data_dict, title="Prediction Input Summary"):
    """Displays the input values in a horizontal table."""
    st.subheader(title)
    display_dict = {key.replace('_', ' ').title(): value for key, value in input_data_dict.items()}
    summary_df = pd.DataFrame([display_dict])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# --- 8. Cached Model Training Functions ---
@st.cache_data(show_spinner=False)
def run_model_comparison(_X_data, _y_data, _models, _preprocessor):
    """Trains multiple models for comparison and returns performance metrics."""
    performance_results = []; progress_bar = st.progress(0, text="Initializing comparison..."); total_models = len(_models)
    X_train, X_test, y_train, y_test = train_test_split(_X_data, _y_data, test_size=0.2, random_state=42)
    for i, (name, model) in enumerate(_models.items()):
        progress_bar.progress((i) / total_models, text=f"Training {name}...")
        pipeline = Pipeline([('preprocessor', _preprocessor), ('regressor', model)])
        try:
            pipeline.fit(X_train, y_train); y_pred = pipeline.predict(X_test)
            mse=mean_squared_error(y_test, y_pred); rmse=np.sqrt(mse); mae=mean_absolute_error(y_test, y_pred); r2=r2_score(y_test, y_pred)
            performance_results.append({'Model': name, 'MSE (Scaled)': mse, 'RMSE (Scaled)': rmse, 'MAE (Scaled)': mae, 'R² (Scaled)': r2, 'Error': None})
        except Exception as e:
             performance_results.append({'Model': name, 'MSE (Scaled)': np.nan, 'RMSE (Scaled)': np.nan, 'MAE (Scaled)': np.nan, 'R² (Scaled)': np.nan, 'Error': str(e)})
    progress_bar.progress(1.0, text="Comparison complete!")
    return pd.DataFrame(performance_results)

# Removed get_simple_prediction_pipeline function

@st.cache_resource(show_spinner=False) # Hide spinner for cached function
def get_prediction_models_dict(_df_full_data, _cat_features, _num_features):
    """
    Trains pipelines for multiple regression models using the full dataset ('All Classes').
    Returns a dictionary where keys are model names and values are fitted pipelines.
    """
    # Define the models to be used for prediction
    prediction_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }

    # Prepare data using the FULL dataset (df_original)
    required_cols = _cat_features + _num_features
    missing = [c for c in required_cols if c not in _df_full_data.columns]
    if missing:
        st.error(f"Error during prediction model setup: Missing columns in df_original: {missing}")
        return {name: None for name in prediction_models.keys()}

    X_train_all = _df_full_data[required_cols]
    y_train_all_scaled = _df_full_data['price_standardized']

    # Define the preprocessor (must match the features used above)
    preprocessor_all = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), _num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), _cat_features)
        ],
        remainder='passthrough'
    )

    trained_pipelines = {}
    model_names = list(prediction_models.keys())

    # Use st.spinner for user feedback during this potentially long initial training
    with st.spinner("Training prediction models (one-time setup)... This might take a minute."):
        for i, (name, model) in enumerate(prediction_models.items()):
            pipeline = Pipeline([
                ('preprocessor', preprocessor_all),
                ('regressor', model)
            ])
            try:
                pipeline.fit(X_train_all, y_train_all_scaled) # Train on the entire dataset
                trained_pipelines[name] = pipeline

                # Store feature names from the first successfully trained pipeline
                if 'processed_feature_names_pred' not in st.session_state:
                     try:
                        ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(_cat_features)
                        num_features_actual = _num_features
                        processed_feature_names = num_features_actual + list(ohe_feature_names)
                        st.session_state['processed_feature_names_pred'] = processed_feature_names
                     except Exception:
                        st.session_state['processed_feature_names_pred'] = None

            except Exception as e:
                st.warning(f"Could not train {name} for prediction: {e}")
                trained_pipelines[name] = None # Store None if training failed

    # st.success("Prediction models ready!") # Can remove this for less noise
    return trained_pipelines

# --- Trigger Training of Prediction Models ---
# Train models using ALL data and features (excluding duration)
categorical_features_for_pred_model = categorical_columns_all.copy()
numerical_features_for_pred_model = numerical_columns_all.copy() # Excludes duration
# Train/Cache all prediction models now
trained_prediction_pipelines = get_prediction_models_dict(
    df_original,
    categorical_features_for_pred_model,
    numerical_features_for_pred_model
)


# --- 9. Main Application Logic (Pages) ---

# =============================================================================
# PAGE 1: DATA OVERVIEW
# =============================================================================
if app_mode == "Data Overview":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"📊 Data Overview ({class_filter})")
    st.markdown("A quick look at the dataset structure and key statistics.")
    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Flights", f"{len(df_display):,}")
    with col2: st.metric("Price Range (₹)", f"{df_display['price'].min():,.0f} - {df_display['price'].max():,.0f}")
    with col3: st.metric("Average Price (₹)", f"{df_display['price'].mean():,.0f}")
    st.markdown("---")
    col_data, col_types = st.columns([3, 1])
    with col_data:
        st.subheader("Data Sample")
        st.dataframe(df_display.drop(columns=['price_standardized', 'duration'], errors='ignore').head(10), use_container_width=True)
    with col_types:
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame(df_display.drop(columns=['price_standardized', 'duration'], errors='ignore').dtypes, columns=['Data Type']).astype(str)
        st.dataframe(dtype_df, use_container_width=True)
    st.markdown("---")
    st.subheader("Price Distribution Analysis")
    col_hist, col_box = st.columns(2)
    with col_hist:
        st.markdown("**Price Histogram (Original ₹)**")
        fig_hist = px.histogram(df_display, x='price', nbins=50, title=f"Frequency of Prices ({class_filter})", labels={'price': 'Price (₹)'})
        fig_hist.update_layout(bargap=0.1); st.plotly_chart(fig_hist, use_container_width=True)
    with col_box:
        st.markdown("**Price Box Plot (Original ₹)**")
        fig_box = px.box(df_display, y='price', title=f"Spread of Prices ({class_filter})", labels={'price': 'Price (₹)'})
        st.plotly_chart(fig_box, use_container_width=True)
    with st.expander("ℹ️ How to Read These Distribution Charts"): st.markdown("...") # Explanation text kept
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 2: PRICE ANALYSIS
# =============================================================================
elif app_mode == "Price Analysis":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"🔍 Price Analysis ({class_filter})")
    st.markdown("Exploring factors affecting flight prices (using original prices in ₹).")
    # Chart 1: Airline vs. Price
    st.subheader("Airline vs. Price"); st.markdown("Comparing price distributions across airlines.")
    fig_airline_box = px.box(df_display, x='airline', y='price', color='airline', title="Price Distribution by Airline", labels={'airline': 'Airline', 'price': 'Price (₹)'})
    fig_airline_box.update_layout(xaxis_tickangle=-45); st.plotly_chart(fig_airline_box, use_container_width=True)
    with st.expander("ℹ️ How to Read (Airline vs. Price)"): st.markdown("...") # Explanation text kept
    st.markdown("---")
    # Chart 2: Airline Market Share
    st.subheader("Airline Market Share"); st.markdown("Percentage of flights operated by each airline.")
    airline_counts = df_display['airline'].value_counts().reset_index(); airline_counts.columns = ['airline', 'count']
    fig_airline_pie = px.pie(airline_counts, names='airline', values='count', title=f'Airline Market Share ({class_filter})', hole=0.3)
    fig_airline_pie.update_traces(textposition='inside', textinfo='percent+label'); st.plotly_chart(fig_airline_pie, use_container_width=True)
    with st.expander("ℹ️ How to Read Pie Chart"): st.markdown("...") # Explanation text kept
    st.markdown("---")
    # Chart 3: Days Left vs. Price
    st.subheader("Price vs. Booking Time")
    st.markdown("**Price vs. Days Before Departure**")
    fig_days_scatter = px.scatter(df_display, x='days_left', y='price', trendline='ols', trendline_color_override='red', title="Price vs. Days Left", labels={'days_left': 'Days Before Departure', 'price': 'Price (₹)'})
    st.plotly_chart(fig_days_scatter, use_container_width=True)
    with st.expander("ℹ️ How to Read (Days Left vs. Price)"): st.markdown("...") # Explanation text kept
    st.markdown("---")
    # Chart 4: Correlation Heatmap
    st.subheader("Feature Correlation Heatmap"); st.markdown("Visualizes linear relationships..."); st.info("ℹ️ Note: Uses Label Encoding...")
    try:
        df_heatmap = df_display.copy(); label_encoders = {}
        for col in df_heatmap.select_dtypes(include=['object']).columns: le = LabelEncoder(); df_heatmap[col] = le.fit_transform(df_heatmap[col].astype(str)); label_encoders[col] = le
        df_heatmap_numeric = df_heatmap.select_dtypes(include=np.number).drop(columns=['price_standardized', 'duration'], errors='ignore')
        if not df_heatmap_numeric.empty:
            corr_matrix = df_heatmap_numeric.corr(); fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, ax=ax_heatmap)
            plt.title('Correlation Matrix (incl. Price)', fontsize=16); st.pyplot(fig_heatmap)
        else: st.warning("No numerical features for heatmap.")
        with st.expander("ℹ️ How to Read Heatmap"): st.markdown("...") # Explanation text kept
    except Exception as e: st.error(f"Could not generate heatmap: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 3: MODEL PERFORMANCE COMPARISON
# =============================================================================
elif app_mode == "Model Performance":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"⚖️ Model Performance Comparison ({class_filter})")
    st.markdown("Comparing models using standardized price prediction. Ranked by R².")
    st.info("ℹ️ Models trained on standardized price. R² comparable regardless of scaling.")
    # Define models
    models_to_compare = {
        'Linear Regression': LinearRegression(), 'Ridge Regression (L2)': Ridge(alpha=1.0),
        'Lasso Regression (L1)': Lasso(alpha=0.001), 'Elastic Net': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
    }
    # Data Prep using df_display
    current_categorical_comp = categorical_columns_all.copy(); current_numerical_comp = numerical_columns_all.copy() # Use global lists (duration excluded)
    if class_filter != 'All Classes' and 'class' in current_categorical_comp: current_categorical_comp.remove('class')
    required_cols = current_categorical_comp + current_numerical_comp
    missing_cols = [col for col in required_cols if col not in df_display.columns]
    if missing_cols: st.error(f"Missing columns: {missing_cols}."); st.stop()
    if df_display.empty or len(df_display) < 50: st.warning(f"Insufficient data ({len(df_display)} rows)."); st.stop()
    X = df_display[required_cols]; y = df_display['price_standardized']
    preprocessor_comp = ColumnTransformer(transformers=[('num', StandardScaler(), current_numerical_comp), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), current_categorical_comp)], remainder='passthrough')
    # Trigger and Display
    if st.button("Compare All Models", type="primary", use_container_width=True):
        performance_df = run_model_comparison(X, y, models_to_compare, preprocessor_comp) # Cached
        st.session_state['performance_df'] = performance_df; st.session_state['comparison_run'] = True
    if st.session_state.get('comparison_run', False):
        if 'performance_df' in st.session_state:
            performance_df = st.session_state['performance_df']
            st.subheader("📊 Performance Metrics (Standardized Scale - Ranked by R²)")
            st.markdown("Lower **MSE**, **RMSE**, **MAE** better. Higher **R²** better.")
            display_df = performance_df.copy(); display_df = display_df.sort_values(by='R² (Scaled)', ascending=False).reset_index(drop=True)
            for col in ['MSE (Scaled)', 'RMSE (Scaled)', 'MAE (Scaled)', 'R² (Scaled)']:
                if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "Error")
            st.dataframe(display_df.drop(columns=['Error'], errors='ignore'), use_container_width=True)
            if 'Error' in performance_df.columns and performance_df['Error'].notna().any(): st.warning("Errors occurred:"); st.dataframe(performance_df[['Model', 'Error']].dropna(), use_container_width=True)
            st.markdown("---"); st.subheader("📈 Visual Performance Comparison")
            col_chart1, col_chart2 = st.columns(2)
            plot_df = performance_df.dropna(subset=['R² (Scaled)', 'RMSE (Scaled)']).sort_values(by='R² (Scaled)', ascending=False)
            if not plot_df.empty:
                 with col_chart1: st.markdown("**R² Score Comparison**"); fig_r2 = px.bar(plot_df, x='Model', y='R² (Scaled)', title='R² Score (Higher is Better)', color='R² (Scaled)', text_auto='.4f', color_continuous_scale='viridis'); min_r2_vis = max(0, plot_df['R² (Scaled)'].min() - 0.05) if pd.notna(plot_df['R² (Scaled)'].min()) else 0; max_r2_vis = min(1, plot_df['R² (Scaled)'].max() + 0.05) if pd.notna(plot_df['R² (Scaled)'].max()) else 1; fig_r2.update_layout(yaxis_range=[min_r2_vis ,max_r2_vis]); fig_r2.update_traces(textangle=0, textposition="outside"); st.plotly_chart(fig_r2, use_container_width=True)
                 with col_chart2: st.markdown("**RMSE Comparison**"); fig_rmse = px.bar(plot_df, x='Model', y='RMSE (Scaled)', title='RMSE (Lower is Better)', color='RMSE (Scaled)', text_auto='.4f', color_continuous_scale='plasma_r'); fig_rmse.update_traces(textangle=0, textposition="outside"); st.plotly_chart(fig_rmse, use_container_width=True)
                 # *** FIXED: Added explanation text ***
                 with st.expander("ℹ️ How to Read Comparison Charts"):
                     st.markdown("""
                        * **R² Score:** How well model explains price variation (closer to 1 is better). RF/GB usually best.
                        * **RMSE:** Average prediction error (scaled). Lower is better. Best models usually have lowest RMSE.
                    """)
            else: st.warning("No valid results to display charts.")
        else: st.warning("Performance data not found.")
    else: st.info("Click 'Compare All Models' button.")
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 4: PREDICT PRICE (Unified Page with Model Selection)
# =============================================================================
elif app_mode == "Predict Price":
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    st.header(f"🎯 Flight Price Prediction")
    st.markdown("Get a price estimate using different regression models trained on **All Classes** data.")
    st.info("ℹ️ Select your flight details (Source/Destination required) and choose a model to predict the price in Rupees (₹).")

    # Check if the trained pipelines dictionary exists and has models
    if not trained_prediction_pipelines or all(p is None for p in trained_prediction_pipelines.values()):
        st.error("Prediction models could not be trained or loaded. Cannot proceed.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # --- Prediction Form ---
    st.subheader("Enter Flight Details for Prediction:")
    with st.form("prediction_form_unified"):
        col_form1, col_form2, col_form3 = st.columns(3)

        # --- Inputs ---
        with col_form1:
            source_city = st.selectbox("Source City*", sorted(df_original['source_city'].unique()), key="pred_source", help="Mandatory: Select the departure city.")
            # *** UPDATED: 'Any' option for Airline ***
            airline = st.selectbox("Airline (Optional)", ["Any"] + sorted(df_original['airline'].unique()), key="pred_airline", help="Select 'Any' to use the typical airline.")
            # *** UPDATED: 'Any' option for Stops ***
            stops = st.selectbox("Stops (Optional)", ["Any"] + sorted(df_original['stops'].unique(), key=lambda x: ("zero", "one", "two_or_more").index(x) if x != "Any" else -1), key="pred_stops", help="Select 'Any' to use the typical number of stops.")

        with col_form2:
            destination_city = st.selectbox("Destination City*", sorted(df_original['destination_city'].unique()), key="pred_dest", help="Mandatory: Select the arrival city.")
            # *** UPDATED: Removed 'All', Default to 'Economy' ***
            class_options = sorted(df_original['class'].unique()) # Get available classes ('Economy', 'Business')
            try:
                # Find the index of 'Economy' to set it as default
                default_class_index = class_options.index('Economy')
            except ValueError:
                default_class_index = 0 # Fallback to first option if 'Economy' isn't present
            flight_class_input = st.selectbox("Class*", class_options, index=default_class_index, key="pred_class", help="Mandatory: Select the flight class.")
            days_left = st.slider("Days Before Departure", min_value=1, max_value=50, value=15, step=1, key="pred_days")

        with col_form3:
            # Dropdown to select the prediction model
            available_models = {name: pipe for name, pipe in trained_prediction_pipelines.items() if pipe is not None}
            if not available_models:
                 st.error("No prediction models available."); st.stop() # Should be caught earlier, but safety check
            model_name_predict = st.selectbox( "Select Prediction Model*", list(available_models.keys()), key="pred_model_select", help="Mandatory: Choose the algorithm.")
            # Duration input removed from UI

        # Hidden inputs (use default/mode) - still needed for model input dictionary
        default_departure_time = df_original['departure_time'].mode()[0]
        default_arrival_time = df_original['arrival_time'].mode()[0]
        # Duration feature is no longer used by the model

        # Submit button
        predict_button = st.form_submit_button("Predict Price", type="primary")

        if predict_button:
            # Validation
            if source_city == destination_city:
                st.error("Source and Destination city cannot be the same.")
            else:
                try:
                    # --- Prepare Input Data ---
                    # Use defaults if user selected "Any"
                    input_airline = df_original['airline'].mode()[0] if airline == "Any" else airline
                    input_stops = df_original['stops'].mode()[0] if stops == "Any" else stops
                    # Use the selected class directly (no 'All' option anymore)
                    input_class = flight_class_input

                    input_dict = {
                        'airline': input_airline,
                        'source_city': source_city,
                        'departure_time': default_departure_time, # Use default
                        'stops': input_stops,
                        'arrival_time': default_arrival_time, # Use default
                        'destination_city': destination_city,
                        'class': input_class,
                        # 'duration' is removed
                        'days_left': days_left
                    }
                    input_df = pd.DataFrame([input_dict])

                    # Select the correct feature columns needed by the model
                    # Use the lists defined globally (excluding duration)
                    input_cols = categorical_features_for_pred_model + numerical_features_for_pred_model
                    # Ensure all required columns are present in the input_df before filtering
                    for col in input_cols:
                        if col not in input_df.columns:
                            # This case should ideally not happen with the defaults, but as a fallback:
                            input_df[col] = 0 # Or use mean/mode if appropriate
                            st.warning(f"Input column '{col}' was missing, using default value. Prediction might be less accurate.")

                    input_df_filtered = input_df[input_cols]


                    # --- Select Trained Model and Predict ---
                    selected_pipeline = available_models[model_name_predict]
                    predicted_price_scaled = selected_pipeline.predict(input_df_filtered)[0]

                    # --- Inverse Transform ---
                    predicted_price_real = target_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

                    # --- Display Prediction ---
                    st.success(f"### Predicted Flight Price ({model_name_predict}): **₹{predicted_price_real:,.2f}**")

                    # --- Display Input Summary (Removed) ---
                    # display_prediction_summary(...) # Removed call

                    # --- Display Context & Confidence (Optional) ---
                    st.markdown("---"); st.subheader("Prediction Context")

                    # Market Comparison (Still useful)
                    compare_class = input_class # Use the actual class used for prediction

                    similar_flights_query = (
                        (df_original['airline'] == input_airline) &
                        (df_original['source_city'] == source_city) &
                        (df_original['destination_city'] == destination_city) &
                        (df_original['class'] == compare_class) & # Compare using the specific class used for prediction
                        (df_original['stops'] == input_stops) &
                        (df_original['days_left'].between(max(1, days_left - 3), days_left + 3))
                    )

                    similar_flights = df_original[similar_flights_query]

                    st.markdown("**Comparison with Similar Historical Flights:**")
                    if not similar_flights.empty:
                        similar_avg = similar_flights['price'].mean(); similar_min = similar_flights['price'].min(); similar_max = similar_flights['price'].max(); similar_count = len(similar_flights)
                        st.markdown(f"* Found **{similar_count}** flights matching key criteria within +/- 3 days ({compare_class}).\n* **Observed Price Range:** ₹{similar_min:,.0f} - ₹{similar_max:,.0f}\n* **Observed Average Price:** ₹{similar_avg:,.0f}\n* **Your Prediction:** **₹{predicted_price_real:,.0f}**")
                        # *** FIXED: Added explanation text ***
                        with st.expander("ℹ️ How to Read Market Comparison"):
                             st.markdown("""
                                Compares prediction to actual prices of similar past flights. Is prediction within range? How does it compare to average?
                            """)
                    else:
                        st.warning("No closely matching flights found in historical data.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.exception(e) # Show full traceback

    st.markdown('</div>', unsafe_allow_html=True) # End page section


# --- 10. Footer/Technical Details ---
st.markdown("---")
st.markdown("##### 🛠️ App Technical Details")
with st.expander("Show Implementation Notes"):
    # *** UPDATED Footer Text ***
    st.markdown(f"""
    * **Data Source:** `Clean_Dataset.csv` loaded locally. Filter (**{class_filter}**) applied for Overview/Analysis pages.
    * **Target Scaling:** `StandardScaler` applied to 'price' *before* training. Predictions inverse-transformed to ₹.
    * **Preprocessing:** `Pipeline` with `ColumnTransformer` (Numerical: `StandardScaler`, Categorical: `OneHotEncoder`). **Duration** feature excluded.
    * **Model Comparison:** Evaluates multiple models, ranked by R² (using filtered data).
    * **Predict Price Page:** Trains multiple models (`LinearRegression`, `Ridge`, `Lasso`, `RandomForest`, `GradientBoosting`) on **All Classes** data (excluding duration). User selects model, inputs details (Source/Dest mandatory, others optional with defaults), and gets prediction in ₹. Class input defaults to 'Economy'. Airline/Stops have 'Any' option. Prediction summary table removed.
    * **Model Caching:** Uses `@st.cache_resource` for pipelines, `@st.cache_data` for data/comparison results.
    * **UI:** Streamlit multi-page app with sidebar, Plotly charts, forms, CSS highlighting. UI spacing improved.
    """)

# --- End of Script ---
