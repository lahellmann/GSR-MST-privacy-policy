import requests
from flask import Flask, request, redirect, session
from urllib.parse import urlencode
import webbrowser
import secrets
import datetime
import pandas as pd
import json

secret_string = secrets.token_hex(16)  # Generate a 32-character hexadecimal string
app = Flask(__name__)
app.secret_key = secret_string


# Replace these with your application's client ID and client secret
client_id = '664b19b2ee885816894611ac'
client_secret = 'JBDqtohJiaC4m6FsOfLJwnU18vPFlXOOdKEWau6m'

# GitHub's OAuth 2.0 endpoints
authorization_endpoint = 'https://api.health.cloud.cardiomood.com/oauth/authorize'
token_endpoint = "https://api.health.cloud.cardiomood.com/oauth/token"

# The redirect URI you registered with your OAuth 2.0 provider
redirect_uri = 'http://localhost:5000/callback'

# Authorization URL
authorization_url = f"{authorization_endpoint}?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code"

@app.route('/unity-data', methods=['POST'])
def receive_unity_data():
    data = request.json  # Assuming Unity sends JSON data
    # Process the received data
    print("Received data from Unity:", data)
    # Example: Merge Unity data with Cardiomood data
    unity_cardiomood_data = {
        "unity_data": data,
        "cardiomood_data": {"example_key": "example_value"}
    }
    return json.dumps(unity_cardiomood_data)

@app.route('/')
def home():
    # Redirect the user to the authorization URL
    return redirect(authorization_url)


@app.route('/callback')
def callback():
    # Get the authorization code from the URL
    authorization_code = request.args.get('code')
    if not authorization_code:
        return 'Authorization code not found in the callback URL.', 400

    # Exchange the authorization code for access and refresh tokens
    tokens = get_tokens(client_id, client_secret, authorization_code, redirect_uri)
    
    if tokens.get('access_token'):
        # Save the refresh token in the session
        session['refresh_token'] = tokens.get('refresh_token')
        
        # Fetch data from Cardiomood API
        access_token = tokens.get("access_token")

        # Fetch raw metrics using the access token
        from_date, to_date = get_date_range()
        raw_stress_data = fetch_raw_metrics(tokens['access_token'], "heart-rate-variability", from_date, to_date)
        print("Raw Stress Data:", raw_stress_data)

        # Save the raw stress data to an Excel file
        if raw_stress_data:
            df = pd.DataFrame(raw_stress_data)
            df.to_excel("raw_stress_data.xlsx", index=False)
            print("Raw stress data saved to raw_stress_data.xlsx")
        else:
            print("No raw stress data to save")
        # Perform further processing or return the fetched data as needed
        return f"Access token: {access_token}, Refresh token saved in session."
    else:
        return "Failed to obtain access token", 400
    
def get_date_range():
    # Get current date and time in UTC+02:00
    current_time_utc = datetime.datetime.utcnow() + datetime.timedelta(hours=2)

    # Set the time to 7 am
    desired_time = current_time_utc.replace(hour=7, minute=0, second=0, microsecond=0)

    # Format the date and time strings
    from_date = desired_time.strftime("%Y-%m-%dT%H:%M:%S.000+02:00")
    to_date = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.000+02:00")
    
    return from_date, to_date

def get_tokens(client_id, client_secret, code, redirect_uri):
    url = token_endpoint
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri
    }
    response = requests.post(url, data=data)
    return response.json()

# Function to fetch raw metrics
def fetch_raw_metrics(access_token, metric, from_date, to_date):
    url = f"https://api.health.cloud.cardiomood.com/v2/raw-metrics/{metric}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    params = {
        "from_date": from_date,
        "to_date": to_date
    }
    response = requests.get(url, headers=headers, params=params)
    try:
        json_response = response.json()
    except ValueError:
        print("Error: Response is not valid JSON")
        return None
    return json_response

# Function to fetch user summaries
def fetch_user_summaries(access_token, date_from, date_to, include_slots=False, per_page=10, page=1):
    url = "https://api.health.cloud.cardiomood.com/user-summaries"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "include_slots": 1 if include_slots else 0,
        "per_page": per_page,
        "page": page
    }
    response = requests.get(url, headers=headers, params=params)
    try:
        json_response = response.json()
    except ValueError:
        print("Error: Response is not valid JSON")
        return None
    return json_response

if __name__ == '__main__':
    # Open the authorization URL in the default web browser
    webbrowser.open('http://localhost:5000')
    
    # Run the Flask web server
    app.run(port=5000)