import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
# Make sure model.pkl and scaler.pkl are in the same directory as app.py
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if files are missing

# Load the original dataset for display
# Make sure 'winequality-red.csv' is in your 'data/' folder
try:
    df = pd.read_csv('data\WineQT.csv')
except FileNotFoundError:
    st.error("Dataset file 'WineQT.csv' not found. Please ensure it's in the 'data/' directory.")
    st.stop()

# --- Required Features --- [cite: 32]

# Title and Description: Clear app title and project description [cite: 33, 34]
st.title("🍷 Wine Quality Prediction App")
st.write("""
This application predicts the quality of red wine based on its physicochemical properties.
It uses a Machine Learning model trained on the Wine Quality Dataset from Kaggle.
""")

# Sidebar Navigation: Organise content with sidebar menus [cite: 34]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

if page == "Data Exploration":
    st.header("Data Exploration")
    st.write("Understand the dataset structure and sample data.")

    # Display dataset overview (shape, columns, data types) 
    st.subheader("Dataset Overview")
    st.write(f"Shape of the dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("Column names and data types:")
    st.dataframe(df.dtypes.rename('Data Type')) # Display data types as a dataframe

    # Show sample data [cite: 35, 36]
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Interactive data filtering options [cite: 37]
    st.subheader("Interactive Data Filtering")
    # Example: Filter by wine quality
    min_quality, max_quality = st.slider(
        "Filter by Quality",
        int(df['quality'].min()), int(df['quality'].max()),
        (int(df['quality'].min()), int(df['quality'].max()))
    )
    filtered_df = df[(df['quality'] >= min_quality) & (df['quality'] <= max_quality)]
    st.write(f"Displaying {len(filtered_df)} rows for quality between {min_quality} and {max_quality}.")
    st.dataframe(filtered_df)

elif page == "Visualizations":
    st.header("Visualizations")
    st.write("Explore relationships and distributions within the dataset.")

    # At least 3 different charts/plots [cite: 39, 40]
    # Interactive visualizations using Streamlit widgets [cite: 41]

    st.subheader("Distribution of Wine Quality")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='quality', data=df, ax=ax1)
    ax1.set_title("Count of Wine Quality Ratings")
    st.pyplot(fig1)

    st.subheader("Alcohol vs. Quality (Box Plot)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='quality', y='alcohol', data=df, ax=ax2)
    ax2.set_title("Alcohol Content by Wine Quality")
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    ax3.set_title("Correlation Matrix of Wine Features")
    st.pyplot(fig3)

    # You could add more interactive plots here, e.g.,
    # Let user select two features for a scatter plot
    st.subheader("Scatter Plot of Features")
    feature_x = st.selectbox("Select X-axis feature", df.columns[:-1]) # Exclude quality
    feature_y = st.selectbox("Select Y-axis feature", df.columns[:-1]) # Exclude quality
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=feature_x, y=feature_y, hue='quality', data=df, ax=ax4)
    ax4.set_title(f"{feature_x} vs. {feature_y} colored by Quality")
    st.pyplot(fig4)


elif page == "Model Prediction":
    st.header("Model Prediction")
    st.write("Enter the physicochemical properties to get a wine quality prediction.")

    # Input widgets for users to enter feature values [cite: 43]
    st.subheader("Enter Wine Properties:")

    # Create input fields for each feature (excluding 'quality')
    input_features = {}
    for column in df.columns[:-1]: # All columns except 'quality'
        # Determine appropriate widget based on data range and type
        min_val = float(df[column].min())
        max_val = float(df[column].max())
        mean_val = float(df[column].mean())

        # Using number_input for more control, allowing decimals
        input_features[column] = st.number_input(
            f"Enter {column.replace('_', ' ').title()}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100.0 # Small step for precision
        )

    if st.button("Predict Wine Quality"):
        # Create a DataFrame from user inputs
        input_df = pd.DataFrame([input_features])

        # Implement error handling for user inputs (e.g., check for valid ranges) 
        # (Already partially handled by min_value/max_value in number_input)
        # You could add more explicit checks here if needed, e.g., if a value is outside a reasonable range

        with st.spinner('Making prediction...'): # Include loading states for long operations 
            try:
                # Scale the input features using the loaded scaler
                input_scaled = scaler.transform(input_df)

                # Make prediction
                prediction = model.predict(input_scaled)
                st.success(f"Predicted Wine Quality: **{int(prediction[0])}**")

                # Prediction confidence/probability (if applicable) [cite: 45]
                if hasattr(model, 'predict_proba'): # Check if model provides probabilities
                    probabilities = model.predict_proba(input_scaled)[0]
                    st.subheader("Prediction Confidence:")
                    prob_df = pd.DataFrame({
                        'Quality': model.classes_,
                        'Probability': probabilities
                    }).set_index('Quality')
                    st.bar_chart(prob_df)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


elif page == "Model Performance":
    st.header("Model Performance")
    st.write("Review the evaluation metrics and performance charts of the trained model.")

    # This part requires re-running the model evaluation or storing the metrics
    # For a real deployment, you'd save these metrics after training and load them here.
    # For this assignment, you can copy-paste the performance metrics you got from your Jupyter notebook.

    st.subheader("Model Evaluation Metrics (Random Forest)")
    st.write("""
    Based on the training and evaluation in the Jupyter Notebook, the Random Forest Classifier
    was selected as the best-performing model.
    """)

    # Display model evaluation metrics [cite: 47]
    st.markdown("""
    **Accuracy on Test Set:** 0.82 (Example value - replace with your actual score)
    """)

    st.subheader("Classification Report (Random Forest - Example)")
    # This would ideally be loaded from a saved report or generated from a test set
    # For a beginner, you can hardcode a representative report or run a small prediction to get one.
    st.code("""
    Example Classification Report:
                  precision    recall  f1-score   support

           3       0.00      0.00      0.00         1
           4       0.00      0.00      0.00        10
           5       0.79      0.88      0.83       130
           6       0.86      0.79      0.83       118
           7       0.79      0.79      0.79        30
           8       0.00      0.00      0.00         1

    avg / total    0.80      0.82      0.81       290
    """, language='python') # Replace with your actual report

    # Confusion matrix or relevant performance charts [cite: 48]
    st.subheader("Confusion Matrix (Random Forest - Example)")
    # To display a confusion matrix, you would need to re-evaluate the model on the test set
    # or save the confusion matrix from training. For simplicity, let's provide a placeholder
    # or you can re-run the prediction here with X_test if you want to generate it live.
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # Ensure X_test and y_test are available for live generation in app.py if desired,
    # or save/load the matrix from training.
    # For now, let's assume you'll generate it in your notebook and describe it.

    # Example of how you *would* generate it if X_test, y_test were available:
    # y_pred = model.predict(scaler.transform(X_test))
    # cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    # fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    # disp.plot(ax=ax_cm, cmap='Blues')
    # ax_cm.set_title('Confusion Matrix')
    # st.pyplot(fig_cm)

    st.write("A confusion matrix would show the counts of correct and incorrect predictions for each quality class.")
    st.write("Refer to the project report for the actual confusion matrix from model training.")

    # Model comparison results [cite: 49]
    st.subheader("Model Comparison")
    st.markdown("""
    During model training, a **Random Forest Classifier** was compared against a **Logistic Regression** model.

    | Model                  | Average Cross-Validation Accuracy | Test Set Accuracy |
    |------------------------|-----------------------------------|-------------------|
    | **Random Forest** | 0.78 (Example)                    | 0.82 (Example)    |
    | Logistic Regression    | 0.60 (Example)                    | 0.63 (Example)    |

    The Random Forest model consistently outperformed Logistic Regression in this specific task.
    """)

# Apply consistent styling and layout 
# Streamlit handles much of this, but you can use st.beta_columns, st.expander, etc.
# Add documentation/help text for users 
st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates a machine learning model deployment using Streamlit.")