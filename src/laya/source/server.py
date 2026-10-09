import laya_mlx as laya
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
from datetime import datetime

# Load the model weights once at startup
print("Loading model weights...")
agent = laya.load("aac6fef/laya-mlx")
print("Model loaded successfully!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    state: Any
    questions: Dict[str, Any]

# In-memory store for IoT device states
device_states = {}

class DeviceData(BaseModel):
    device_id: str
    temperature: float
    vibration: float
    local_prediction: str
    status: str

@app.post("/predict")
async def predict(request: InferenceRequest):
    result = agent.predict(request.state, request.questions)
    return result

@app.post("/api/devices/{device_id}")
async def update_device(device_id: str, data: DeviceData):
    device_states[device_id] = {
        **data.model_dump(),
        "last_updated": datetime.now().isoformat()
    }
    
    # Use Laya as the Decision Layer
    state_desc = f"Device {device_id} is reporting {data.temperature}C temp, {data.vibration}g vibration. Local ML suggests {data.local_prediction}."
    
    questions = {
        "trigger_maintenance": {
            "type": "noul",
            "instructions": "Should maintenance be triggered because of elevated metrics or anomaly?"
        },
        "shutdown_device": {
            "type": "noul",
            "instructions": "Should the device be shut down immediately due to a critical anomaly?"
        }
    }
    
    # We call Laya prediction
    try:
        decision = agent.predict({"state": state_desc}, questions)
        device_states[device_id]["laya_decision"] = decision
        device_states[device_id]["laya_trace"] = {
            "state_provided": state_desc,
            "questions_asked": questions
        }
    except Exception as e:
        device_states[device_id]["laya_decision"] = {"error": str(e)}
        device_states[device_id]["laya_trace"] = None
        
    return device_states[device_id]

@app.get("/api/devices")
async def get_devices():
    return list(device_states.values())
