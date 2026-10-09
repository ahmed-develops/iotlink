import laya_mlx as laya
import json

agent = laya.load("aac6fef/laya-mlx")
state_desc = "Device IoT-Pump-01 is reporting 45.0C temp, 0.1g vibration. Local ML suggests Normal."
questions = {
    "trigger_maintenance": {
        "type": "noul",
        "instructions": "Should maintenance be triggered based on this device state? Reply ONLY with 'Yes' or 'No'."
    },
    "shutdown_device": {
        "type": "noul",
        "instructions": "Should the device be immediately shut down? Reply ONLY with 'Yes' or 'No'."
    }
}
try:
    decision = agent.predict({"state": state_desc}, questions)
    print("NORMAL RAW OUTPUT:", decision)
except Exception as e:
    print("ERROR:", e)
