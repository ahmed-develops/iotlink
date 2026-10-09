import laya_mlx as laya
import json

agent = laya.load("aac6fef/laya-mlx")
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

states = {
    "Normal": "Device IoT-Pump-01 is reporting 45.0C temp, 0.1g vibration. Local ML suggests Normal.",
    "Warning": "Device IoT-Pump-01 is reporting 65.0C temp, 0.6g vibration. Local ML suggests Warning: Elevated metrics.",
    "Critical": "Device IoT-Pump-01 is reporting 85.5C temp, 1.2g vibration. Local ML suggests Critical Anomaly Detected."
}

for name, st in states.items():
    print(f"\n--- {name} ---")
    decision = agent.predict({"state": st}, questions)
    print("Maint noul:", decision["answers"]["trigger_maintenance"]["noul"])
    print("Shut  noul:", decision["answers"]["shutdown_device"]["noul"])
