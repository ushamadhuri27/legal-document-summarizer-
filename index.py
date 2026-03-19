from flask import Flask, jsonify, request, render_template, send_from_directory
import json
import uuid
import random
import joblib
import os
import gradio_client
from googletrans import Translator
from geopy.geocoders import Nominatim
from flask_cors import CORS
import re
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Configure CORS for all routes
CORS(app, 
     resources={
         r"/*": {
             "origins": ["http://localhost:5173"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True
         }
     })

# Initialize models as None
clientModel = None
clientLabelEncoder = None
caseTokenizer = None
caseModel = None
labelEncoder = None

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to load models if they exist
try:
    # Load client classification model components
    model_path = os.path.join(BASE_DIR, 'client.joblib')
    label_encoder_path = os.path.join(BASE_DIR, 'client_label_encoder.joblib')
    
    clientModel = joblib.load(model_path)
    clientLabelEncoder = joblib.load(label_encoder_path)
    print("Successfully loaded client classification model components")
except Exception as e:
    print(f"Warning: Client classification model not found or error loading: {e}")

try:
    # Load the case classification model components
    model_path = os.path.join(BASE_DIR, 'case_legalbert_model')
    caseTokenizer = AutoTokenizer.from_pretrained(model_path)
    caseModel = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    labelEncoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    print("Successfully loaded case classification model components")
except Exception as e:
    print(f"Warning: Case classification model not found or error loading: {e}")

def translateChecks(text: str) -> str:
    translator = Translator()

    print(text)
    detected_lang = translator.detect(text).lang

    if detected_lang != "en":
        
        translated_text = translator.translate(text, dest="en")
        return translated_text
    return text    

def getMatchScore(lawyerObj, clientReqObj):
    totalPoints = 0
    caseTypePoints = 0
    for cCaseType in clientReqObj["caseType"]:
        for lCaseType in lawyerObj["speciality"]:
            if lCaseType == cCaseType:
                if caseTypePoints == 0:
                    caseTypePoints += 10
                else:
                    caseTypePoints += 5

    totalPoints += caseTypePoints

    
    languagePoints = 0
    for cLang in clientReqObj["languages"]:
        for lLang in lawyerObj["languages"]:
            if lLang == cLang:
                if languagePoints == 0:
                    languagePoints += 10
                else:
                    languagePoints += 2

    totalPoints += languagePoints

    
    budgetPoints = 0
    a = max(clientReqObj["budget"], lawyerObj["price"])
    b = min(clientReqObj["budget"], lawyerObj["price"])
    diff = a - b
    if diff < 250:
        diff = int(diff / 250)
        budgetPoints += diff

    totalPoints += budgetPoints

    
    locationPoint = 0
    if clientReqObj["location"] == lawyerObj["location"]:
        locationPoint += 10
    totalPoints += locationPoint

    
    totalPoints += lawyerObj["rating"] * 2

    
    return totalPoints

def sortFunction(t):
    return t[0]

def recommendedLawyers(clientReqObj) -> list:
    try:
        # Load lawyers from the correct path
        lawyers_path = os.path.join(BASE_DIR, 'lawyers.json')
        with open(lawyers_path, 'r') as f:
            lawyers = json.load(f)
        
        lawyerList = []
        for lawyer in lawyers:
            score = getMatchScore(lawyer, clientReqObj)
            lawyerList.append((score, lawyer))
        
        lawyerList = sorted(lawyerList, key=sortFunction, reverse=True)
        finalList = [obj for (s, obj) in lawyerList]
        
        return finalList  # Return all lawyers without limit
    except Exception as e:
        print(f"Error in recommendedLawyers: {e}")
        return []

def findLocation(latitude: float, longitude: float) -> str:
    try:
        # Only try to get location if coordinates are not 0,0
        if latitude == 0 and longitude == 0:
            return "General"
            
        geolocator = Nominatim(user_agent="my-app", timeout=10)  # Increased timeout
        location = geolocator.reverse(f"{latitude}, {longitude}")
        
        if location is None:
            return "General"
            
        address = location.raw.get("address", {})
        return address.get("city_district", "General")
    except Exception as e:
        print(f"Error finding location: {e}")
        return "General"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Fixed regex pattern
    return text

def getCaseType(query: str) -> [str]:
    if caseModel is None or caseTokenizer is None or labelEncoder is None:
        print("Case classification model not available")
        return ["General"]  # Return default case type if model is not available
    
    try:
        # Clean and preprocess the text
        cleaned_text = preprocess_text(query)
        
        # Tokenize the input
        inputs = caseTokenizer(
            [cleaned_text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='tf'
        )
        
        # Get model predictions
        logits = caseModel(inputs)[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        # Filter predictions with confidence > 8%
        predictions = []
        for idx in top3_idx:
            confidence = probs[idx]
            if confidence > 0.08:
                label = labelEncoder.inverse_transform([idx])[0]
                predictions.append(label)
                print(f"Predicted case type: {label} with confidence: {confidence:.2%}")
        
        return predictions if predictions else ["General"]
        
    except Exception as e:
        print(f"Error in case classification: {e}")
        return ["General"]

def getClientType(query: str) -> [str]:
    if clientModel is None or clientLabelEncoder is None:
        print("Client classification model not available")
        return ["Individual"]  # Return default client type if model is not available
    
    try:
        # Clean and preprocess the text
        cleaned_text = preprocess_text(query)
        
        # Get predictions with probabilities
        predictions = clientModel.predict_proba([cleaned_text])[0]
        
        # Get top predictions with confidence > 8%
        top_values = []
        labels = clientLabelEncoder.classes_
        
        for label, probability in zip(labels, predictions):
            if probability > 0.08:
                if len(top_values) < 3:
                    top_values.append([probability, label])
                else:
                    tops = [i[0] for i in top_values]
                    min_value = min(tops)
                    if probability > min_value:
                        min_index = tops.index(min_value)
                        top_values[min_index] = [probability, label]
        
        # Sort by probability and get labels
        top_values.sort(reverse=True)
        predictions = [label for _, label in top_values]
        
        if predictions:
            print(f"Predicted client types: {predictions}")
            return predictions
        else:
            return ["Individual"]
            
    except Exception as e:
        print(f"Error in client classification: {e}")
        return ["Individual"]

@app.route("/")
def hi():
    return render_template('index.html')

@app.route("/api/rate")
def rate():
    r = request.json.get("rate")
    r = float(r)

    if r > 5:
        return "Error", 400

    return jsonify("ok")

@app.route("/api/query", methods=["POST", "GET"])
def query():
    try:
        q = request.json.get("query", "")
        # Convert to string if it's a Translated object
        if hasattr(q, 'text'):
            q = q.text
        elif not isinstance(q, str):
            q = str(q)
            
        q = translateChecks(q)

        try:
            cases = getCaseType(q)
            print(f"Detected case types: {cases}")  # Debug log
        except Exception as e:
            print(f"Error in case classification: {e}")
            cases = ["Divorce"]  # Default to Divorce if error occurs

        try:
            clientType = getClientType(q)
        except Exception as e:
            print(f"Error in client classification: {e}")
            clientType = "Individual"  # Default to Individual if error occurs

        lat = request.json.get("latitude", 0)
        lon = request.json.get("longitude", 0)
        state = findLocation(float(lat), float(lon))

        # Load all lawyers
        with open(os.path.join(BASE_DIR, 'lawyers.json'), 'r') as f:
            all_lawyers = json.load(f)

        # Filter lawyers based on the detected case types
        matching_lawyers = []
        for lawyer in all_lawyers:
            # Check if lawyer's speciality matches any of the detected case types
            if any(case.lower() in [spec.lower() for spec in lawyer.get('speciality', [])] for case in cases):
                matching_lawyers.append(lawyer)

        if matching_lawyers:
            # Sort matching lawyers by experience and rating
            matching_lawyers.sort(key=lambda x: (x.get('experience', 0), x.get('rating', 0)), reverse=True)
            return jsonify(matching_lawyers)
        else:
            # If no matching lawyers found, return all lawyers
            return jsonify(all_lawyers)

    except Exception as e:
        print(f"Error in query endpoint: {e}")
        # Return all lawyers as fallback
        try:
            with open(os.path.join(BASE_DIR, 'lawyers.json'), 'r') as f:
                all_lawyers = json.load(f)
            return jsonify(all_lawyers)
        except Exception as fallback_error:
            print(f"Error in fallback: {fallback_error}")
            return jsonify([]), 500

@app.route("/api/form", methods=["POST", "GET"])
def form():
    try:
        l = {
            "name": str(request.json.get("name")),
            "id": str(uuid.uuid4()),
            "experience": int(request.json.get("experience")),
            "speciality": (request.json.get("speciality")),
            "location": str(request.json.get("location")),
            "clientType": str(request.json.get("clientType")),
            "rating": random.uniform(0.0, 5.0),
            "jurisdiction": str(request.json.get("jurisdiction")),
            "price": float(request.json.get("price")),
            "avgDaysOfCompletion": int(request.json.get("avgDaysOfCompletion")),
            "languages": (request.json.get("languages")),
            "gender": str(request.json.get("gender")),
        }

        with open("lawyers.json", "r") as file:
            data = json.load(file)

        data.append(l)
        print(data[-1])
        with open("lawyers.json", "w") as file:
            json.dump(data, file, indent=4)

        return "ok"
    except:
        return "", 400

# Add route to serve lawyers.json
@app.route("/lawyers.json")
def get_lawyers():
    try:
        # Load lawyers from the correct path
        lawyers_path = os.path.join(BASE_DIR, 'lawyers.json')
        with open(lawyers_path, 'r') as f:
            lawyers = json.load(f)
        return jsonify(lawyers)
    except Exception as e:
        print(f"Error serving lawyers.json: {e}")
        return jsonify([]), 500

def load_users():
    try:
        users_path = os.path.join(BASE_DIR, 'data', 'users.json')
        with open(users_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return {"clients": [], "lawyers": []}

def save_users(users_data):
    try:
        users_path = os.path.join(BASE_DIR, 'data', 'users.json')
        with open(users_path, 'w') as f:
            json.dump(users_data, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")

@app.route('/api/auth/client', methods=['POST'])
def client_auth():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        users_data = load_users()
        client = next((c for c in users_data['clients'] if c['email'] == email and c['password'] == password), None)
        
        if client:
            # Log client login
            print("\n=== Client Login ===")
            print(f"Client Name: {client['name']}")
            print(f"Client ID: {client['id']}")
            print(f"Client Email: {client['email']}")
            print("===================\n")
            
            return jsonify({
                'id': client['id'],
                'name': client['name'],
                'email': client['email']
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"Error in client authentication: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/lawyer', methods=['POST'])
def lawyer_auth():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()

        # Log authentication attempt
        print("\n=== Lawyer Login Attempt ===")
        print(f"Email: {email}")
        print("===========================\n")

        # Check lawyers.json for the email
        lawyers_path = os.path.join(BASE_DIR, 'lawyers.json')
        with open(lawyers_path, 'r') as f:
            lawyers = json.load(f)
        
        # Find lawyer in lawyers.json
        lawyer = next((l for l in lawyers if l.get('email', '').strip() == email), None)
        
        if not lawyer:
            print("\n=== Lawyer Login Failed ===")
            print("Lawyer not found in lawyers list")
            print("===========================\n")
            return jsonify({'error': 'Invalid credentials'}), 401

        # Load cases to check if there are any cases for this lawyer
        cases_data = load_cases()
        lawyer_cases = [c for c in cases_data['cases'] if c['lawyerId'] == lawyer['id']]

        # Log successful login
        print("\n=== Lawyer Login Success ===")
        print(f"Lawyer Name: {lawyer['name']}")
        print(f"Lawyer ID: {lawyer['id']}")
        print(f"Lawyer Email: {lawyer['email']}")
        print("===========================\n")
        
        # Return lawyer data with cases if any
        lawyer_data = {
            **lawyer,  # Include all data from lawyers.json
            'cases': lawyer_cases  # Include cases from cases.json
        }
        return jsonify(lawyer_data)

    except Exception as e:
        print(f"Error in lawyer authentication: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def load_cases():
    try:
        cases_path = os.path.join(BASE_DIR, 'data', 'cases.json')
        with open(cases_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading cases: {e}")
        return {"cases": []}

def save_cases(cases_data):
    try:
        cases_path = os.path.join(BASE_DIR, 'data', 'cases.json')
        with open(cases_path, 'w') as f:
            json.dump(cases_data, f, indent=2)
    except Exception as e:
        print(f"Error saving cases: {e}")

@app.route('/api/cases', methods=['GET'])
def get_cases():
    try:
        lawyer_id = request.args.get('lawyerId')
        client_id = request.args.get('clientId')
        
        cases_data = load_cases()
        
        if lawyer_id:
            # Get cases for lawyer with full details
            cases = [c for c in cases_data['cases'] if c['lawyerId'] == lawyer_id]
            # Sort cases by creation date (newest first)
            cases.sort(key=lambda x: x['createdAt'], reverse=True)
        elif client_id:
            # Get cases for client with full details
            cases = [c for c in cases_data['cases'] if c['clientId'] == client_id]
            # Sort cases by creation date (newest first)
            cases.sort(key=lambda x: x['createdAt'], reverse=True)
        else:
            cases = cases_data['cases']
        
        return jsonify({'cases': cases})
    except Exception as e:
        print(f"Error getting cases: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cases', methods=['POST'])
def create_case():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['lawyerId', 'clientId', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Load users to get client info
        users_data = load_users()
        client = next((c for c in users_data['clients'] if c['id'] == data['clientId']), None)

        # Load lawyers to get lawyer info
        lawyers_path = os.path.join(BASE_DIR, 'lawyers.json')
        with open(lawyers_path, 'r') as f:
            lawyers_list = json.load(f)
        lawyer = next((l for l in lawyers_list if l['id'] == data['lawyerId']), None)

        if not client or not lawyer:
            return jsonify({'error': 'Invalid client or lawyer ID'}), 400

        # Log case submission details
        print("\n=== New Case Submission ===")
        print(f"Client Name: {client['name']}")
        print(f"Client ID: {client['id']}")
        print(f"Lawyer Name: {lawyer['name']}")
        print(f"Lawyer ID: {lawyer['id']}")
        print(f"Case Description: {data['description']}")
        print("=========================\n")

        # Generate a new case ID
        new_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Create case object with full details including logged information
        case = {
            'id': new_id,
            'lawyerId': data['lawyerId'],
            'clientId': data['clientId'],
            'clientName': client['name'],
            'clientEmail': client['email'],
            'lawyerName': lawyer['name'],
            'lawyerEmail': lawyer['email'],
            'description': data['description'],
            'status': 'pending',
            'createdAt': current_time,
            'updatedAt': None,
            'lawyerResponse': None,
            'caseType': data.get('caseType', 'General'),
            'priority': data.get('priority', 'Medium'),
            'budget': data.get('budget', 0),
            'logHistory': [
                {
                    'timestamp': current_time,
                    'event': 'case_created',
                    'details': {
                        'clientName': client['name'],
                        'clientId': client['id'],
                        'lawyerName': lawyer['name'],
                        'lawyerId': lawyer['id'],
                        'description': data['description']
                    }
                }
            ]
        }
        
        # Save to cases.json
        cases_data = load_cases()
        cases_data['cases'].append(case)
        save_cases(cases_data)

        # Update user records with full case details
        client_case = {
            'id': new_id,
            'lawyerId': data['lawyerId'],
            'lawyerName': lawyer['name'],
            'status': 'pending',
            'description': data['description'],
            'createdAt': current_time,
            'caseType': data.get('caseType', 'General'),
            'priority': data.get('priority', 'Medium'),
            'budget': data.get('budget', 0)
        }

        lawyer_case = {
            'id': new_id,
            'clientId': data['clientId'],
            'clientName': client['name'],
            'status': 'pending',
            'description': data['description'],
            'createdAt': current_time,
            'caseType': data.get('caseType', 'General'),
            'priority': data.get('priority', 'Medium'),
            'budget': data.get('budget', 0)
        }

        # Update client's cases
        if 'cases' not in client:
            client['cases'] = []
        client['cases'].append(client_case)

        # Update lawyer's cases in users.json if present
        if lawyer and 'cases' in lawyer:
            # This only updates lawyers in users.json, not lawyers.json
            pass
        for l in users_data.get('lawyers', []):
            if l['id'] == lawyer['id']:
                if 'cases' not in l:
                    l['cases'] = []
                l['cases'].append(lawyer_case)

        save_users(users_data)
        
        return jsonify(case), 201
    except Exception as e:
        print(f"Error creating case: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cases/<case_id>', methods=['PUT'])
def update_case(case_id):
    try:
        data = request.get_json()
        cases_data = load_cases()
        users_data = load_users()
        
        case = next((c for c in cases_data['cases'] if c['id'] == case_id), None)
        if not case:
            return jsonify({'error': 'Case not found'}), 404
        
        # Update case status and lawyer response
        case['status'] = data['status']
        case['lawyerResponse'] = data.get('lawyerResponse')
        case['updatedAt'] = datetime.now().isoformat()
        
        # Update case status in user records
        for client in users_data['clients']:
            for client_case in client.get('cases', []):
                if client_case['id'] == case_id:
                    client_case['status'] = data['status']
                    client_case['lawyerResponse'] = data.get('lawyerResponse')
                    client_case['updatedAt'] = datetime.now().isoformat()
                    break

        for lawyer in users_data['lawyers']:
            for lawyer_case in lawyer.get('cases', []):
                if lawyer_case['id'] == case_id:
                    lawyer_case['status'] = data['status']
                    lawyer_case['lawyerResponse'] = data.get('lawyerResponse')
                    lawyer_case['updatedAt'] = datetime.now().isoformat()
                    break
        
        save_cases(cases_data)
        save_users(users_data)
        
        return jsonify(case)
    except Exception as e:
        print(f"Error updating case: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/lawyers/<lawyer_id>', methods=['GET'])
def get_lawyer(lawyer_id):
    try:
        # Load lawyers from the correct path
        lawyers_path = os.path.join(BASE_DIR, 'lawyers.json')
        with open(lawyers_path, 'r') as f:
            lawyers = json.load(f)
        
        # Find the lawyer with matching ID
        lawyer = next((l for l in lawyers if l.get('id') == lawyer_id), None)
        
        if lawyer:
            # Log lawyer view
            print("\n=== Lawyer Information Viewed ===")
            print(f"Lawyer Name: {lawyer['name']}")
            print(f"Lawyer ID: {lawyer['id']}")
            print(f"Specialization: {', '.join(lawyer.get('speciality', []))}")
            print("===============================\n")
            
            return jsonify(lawyer)
        else:
            return jsonify({'error': 'Lawyer not found'}), 404
            
    except Exception as e:
        print(f"Error getting lawyer: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
