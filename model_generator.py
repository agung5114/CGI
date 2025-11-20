import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

def train_and_save_model():
    csv_file = 'final_symptoms_to_disease.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Please ensure the CSV is in the same directory.")
        return

    print(f"Loading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        
        # Clean data
        df.dropna(subset=['symptom_text', 'diseases'], inplace=True)
        X_text = df['symptom_text'].values
        y = df['diseases'].values
        unique_diseases = sorted(list(set(y)))
        
        print(f"Training model on {len(df)} records (using TF-IDF + Naive Bayes)...")
        
        # Create Pipeline (TF-IDF -> Naive Bayes)
        # This is fast, lightweight, and robust for text classification
        pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
        pipeline.fit(X_text, y)
        
        # Prepare dictionary for pickle
        # 'type': 'tfidf' is CRITICAL - it tells cgi_app.py to NOT use BERT encoding for this model
        model_data = {
            'model': pipeline,
            'type': 'tfidf', 
            'classes': unique_diseases,
            'training_sample': df.head()
        }
        
        # Save to disk
        output_file = 'disease_model.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"✅ Success! Model saved to '{output_file}'.")
        print("You can now run 'streamlit run cgi_app.py' and the error will be gone.")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()