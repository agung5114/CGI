Claim Guard Intelligence (CGI)

Enterprise-Grade Healthcare Fraud Defense System

CGI is a state-of-the-art forensic analytics platform designed to detect healthcare fraud, validate clinical consistency, and verify biometric authenticity. It leverages multi-modal Artificial Intelligence—combining Graph Network Analysis, NLP, and Computer Vision—to provide a holistic risk assessment for insurance claims.

🚀 Features

🛡️ Executive Dashboard: Real-time tracking of claims, fraud metrics, and financial savings.

🕸️ Graph Intelligence: Physics-based network visualization to detect collusion rings.

🧠 AI Semantic Matching: NLP-powered clinical note validation and semantic fraud quantification.

✍️ Computer Vision Verification: Signature verification using CLIP embeddings to detect forgeries.

🔒 Enterprise Security: Role-Based Access Control (RBAC) and immutable Audit Logs.

🛠️ Installation

Prerequisites

Python 3.9 or higher

pip (Python Package Installer)

Git

1. Clone the Repository

git clone [https://github.com/your-org/cgi-pro.git](https://github.com/your-org/cgi-pro.git)
cd cgi-pro


2. Create a Virtual Environment

It is recommended to run this application in an isolated environment.

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


Note: If you do not have a requirements.txt, create one with the following content:

flask
pandas
numpy
networkx
scikit-learn
sentence-transformers
pillow
gunicorn


⚙️ Configuration & Setup

1. Initialize Models

The application relies on trained AI models.

Ensure final_symptoms_to_disease.csv is in the root directory.

Run the training script once to generate the optimized pickle file:

python train_model.py


This will create disease_model.pkl.

2. Download Semantic Models (First Run)

The app uses Hugging Face's clip-ViT-B-32 for signature verification. This model (~300MB) will automatically download to the local_clip_model/ folder on the first run. Ensure you have a stable internet connection for the initial launch.

▶️ Running the Application

Development Mode

For local testing and development:

python flask_cgi_app.py


Access the app at: http://127.0.0.1:5000

Note: The camera feature works on localhost without SSL.

Production Deployment

For a production environment, do not use the built-in Flask server. Use a WSGI server like Gunicorn.

Standard Run:

gunicorn --bind 0.0.0.0:8000 flask_cgi_app:app


With Workers (Scaling):

gunicorn --workers 4 --bind 0.0.0.0:8000 flask_cgi_app:app


SSL/HTTPS Requirement (Important)

The Biometric Verification (Camera) feature requires a secure context (HTTPS) to work on non-localhost domains.

Nginx Reverse Proxy: Set up Nginx to handle SSL termination and forward requests to Gunicorn.

Certbot: Use Let's Encrypt to generate a free SSL certificate for your domain.

📂 Project Structure

cgi-pro/
├── flask_cgi_app.py       # Main Application Entry Point
├── train_model.py         # Script to retrain/reset the Disease Model
├── final_symptoms_to_disease.csv # Training Data
├── disease_model.pkl      # Serialized ML Model (Generated)
├── local_clip_model/      # Cached Hugging Face Models
├── requirements.txt       # Python Dependencies
└── README.md              # This file


🛡️ Security Notes

API Keys: The "System Settings" page is a simulation. In a real deployment, replace the mock API keys in flask_cgi_app.py with environment variables (e.g., os.environ.get('DB_KEY')).

Data Privacy: The app processes PII (Simulated). Ensure your deployment complies with HIPAA/GDPR regulations if connecting to real data sources.

🤝 Support

For issues or feature requests, please contact the CGI Development Team.