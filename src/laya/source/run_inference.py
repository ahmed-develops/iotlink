import json
import urllib.request
import urllib.error

# The payload representing state and questions
payload = {
    "state": {"state": "The user wants to upgrade their subscription"},
    "questions": {
        "route_to_billing": {
            "type": "noul",
            "instructions": "Does the user want to upgrade or access billing?"
        },
        "route_to_support": {
            "type": "noul",
            "instructions": "Does the user want to contact customer support and also access billing?"
        }
    }
}

# Send request to the local FastAPI server
url = "http://localhost:8000/predict"
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(json.dumps(result, indent=2))
except urllib.error.URLError as e:
    print(f"Failed to connect to the server. Is it running? Error: {e}")