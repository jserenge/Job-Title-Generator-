import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="HR Job Title Predictor",
    page_icon="👔",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stTextInput > label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .stTextArea > label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .prediction {
            padding: 2rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Load the saved models
@st.cache_resource
def load_models():
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    return rf_model, tfidf_vectorizer, label_encoder

# Preprocess text function
def preprocess_text(text):
    # Add your text preprocessing steps here
    # For example: lowercase, remove special characters, etc.
    return text.lower().strip()

# Main function to run the Streamlit app
def main():
    try:
        # Load models
        rf_model, tfidf_vectorizer, label_encoder = load_models()
        
        # Header
        st.title("🎯 HR Job Title Predictor")
        st.markdown("""
        This tool predicts job titles based on skills, responsibilities, and job descriptions. 
        Enter the details below to get a prediction.
        """)
        
        # Create three columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job Details")
            skills = st.text_area(
                "Skills Required",
                placeholder="Enter key skills (e.g., Python, Project Management, Leadership)",
                height=150
            )
            
            responsibilities = st.text_area(
                "Job Responsibilities",
                placeholder="Enter main job responsibilities",
                height=150
            )

        with col2:
            job_description = st.text_area(
                "Job Description",
                placeholder="Enter detailed job description",
                height=350
            )
        
        # Add a predict button
        if st.button("Predict Job Title", type="primary"):
            if skills and responsibilities and job_description:
                # Combine and preprocess the text
                combined_text = f"{job_description} {responsibilities} {skills}"
                processed_text = preprocess_text(combined_text)
                
                # Transform the text using TF-IDF
                text_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()
                
                # Make prediction
                prediction = rf_model.predict(text_tfidf)
                predicted_title = label_encoder.inverse_transform(prediction)[0]
                
                # Display prediction with styling
                st.markdown("### 🎉 Prediction Results")
                st.markdown(
                    f"""
                    <div class="prediction">
                        <h4>Predicted Job Title:</h4>
                        <h2 style="color: #1f77b4;">{predicted_title}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Add confidence information
                proba = rf_model.predict_proba(text_tfidf)
                top_3_indices = proba[0].argsort()[-3:][::-1]
                top_3_titles = label_encoder.inverse_transform(top_3_indices)
                top_3_probs = proba[0][top_3_indices]
                
                st.markdown("#### Alternative Suggestions:")
                for title, prob in zip(top_3_titles, top_3_probs):
                    st.markdown(f"- {title} *(Confidence: {prob:.1%})*")
                
            else:
                st.error("Please fill in all fields before predicting.")
        
        # Add information about the model
        with st.expander("ℹ️ About this Predictor"):
            st.markdown("""
            This job title predictor uses a Random Forest model trained on 300,000 job listings. 
            It analyzes the following features to make predictions:
            - Required skills and competencies
            - Job responsibilities and duties
            - Detailed job description
            
            The model uses natural language processing techniques to understand the relationship 
            between job requirements and corresponding titles.
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown("""
        Please make sure all required model files are in the same directory:
        - rf_model.pkl
        - tfidf_vectorizer.pkl
        - label_encoder.pkl
        """)

if __name__ == "__main__":
    main()