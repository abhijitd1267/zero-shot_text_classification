# app.py
import streamlit as st
import matplotlib.pyplot as plt
from zero_shot_classifier import ZeroShotTextClassifier

# Initialize the classifier (runs once when the app starts)
@st.cache_resource
def init_classifier():
    return ZeroShotTextClassifier()

classifier = init_classifier()

# Title and layout
st.title("Zero-Shot Text Classification")

# Input fields at the top (always visible)
text_input = st.text_area("Text to Classify", height=100, placeholder="Enter text to classify...")
labels_input = st.text_input("Candidate Labels (comma-separated)", placeholder="e.g., sports, politics, technology")
multi_label = st.checkbox("Allow multiple labels")

# Classify button
if st.button("Classify Text"):
    if not text_input or not labels_input:
        st.error("Please enter text and at least one label.")
    else:
        # Prepare data
        candidate_labels = [label.strip() for label in labels_input.split(",") if label.strip()]
        
        try:
            with st.spinner("Classifying..."):
                # Perform classification
                results = classifier.classify(
                    texts=text_input,
                    candidate_labels=candidate_labels,
                    multi_label=multi_label
                )
                
                # Create columns for results (60% for text, 40% for chart)
                results_col, chart_col = st.columns([6, 4])
                
                # Display results in left column
                with results_col:
                    st.subheader("Results")
                    st.write(f"**Text:** {results['sequence']}")
                    st.write("**Label Scores:**")
                    for label, score in zip(results['labels'], results['scores']):
                        percentage = score * 100
                        st.write(f"- {label}: {percentage:.1f}%")
                
                # Create pie chart in right column
                with chart_col:
                    st.subheader("Score Distribution")
                    
                    # Prepare data for pie chart
                    labels = results['labels']
                    sizes = [score * 100 for score in results['scores']]
                    
                    # Create figure with larger size
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Custom colors and styling
                    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
                    wedgeprops = {'linewidth': 1, 'edgecolor': 'white'}
                    textprops = {'fontsize': 20}  # Increased from 8 to 12
                    
                    # Create pie chart with better formatting
                    patches, texts, autotexts = ax.pie(
                        sizes,
                        labels=labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors[:len(labels)],
                        wedgeprops=wedgeprops,
                        textprops=textprops,
                        pctdistance=0.8
                    )
                    
                    # Improve label positioning and legibility with larger text
                    plt.setp(autotexts, size=20, weight="bold", color="white")  # Increased from 8 to 12
                    plt.setp(texts, size=20)  # Increased from 8 to 12
                    
                    # Equal aspect ratio ensures pie is drawn as a circle
                    ax.axis('equal')
                    
                    # Add a bit of padding around the chart
                    plt.tight_layout()
                    
                    # Display the pie chart
                    st.pyplot(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

# Optional: Add some styling or info
st.sidebar.write("Use this app to classify text into predefined categories using a zero-shot learning model.")