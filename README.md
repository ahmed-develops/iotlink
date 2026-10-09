# IoT Federated Learning + Laya Semantic Routing

A next-generation IoT decision system that demonstrates **Federated Machine Learning (FedML)** at the edge combined with the **Laya AI Semantic Router** in the cloud. 

This project simulates how to build a highly responsive, zero-hallucination industrial IoT dashboard that preserves data privacy by keeping raw data on the device, while executing centralized, context-aware decisions.

## Architecture Overview

1. **The Edge (Federated Learning):** Simulated IoT devices (Pumps, Turbines) run local Logistic Regression models directly in the browser. They use Gradient Descent to learn normal operating baselines. Every few cycles, their mathematical weights are synced centrally using the **Federated Averaging (FedAvg)** algorithm.
2. **Central Intelligence (Laya AI):** Instead of streaming heavy raw data (like 100Hz vibration numbers) to a cloud, devices only send a lightweight semantic payload when an anomaly occurs (e.g., *"Critical Anomaly Detected"*). The centralized **Laya-RL Agent** acts as a Semantic Router, taking this payload and calculating a mathematical confidence score (`noul`) against operational instructions to trigger **Maintenance** or **Shutdown** without the risk of LLM hallucinations.

---

## Getting Started

### Prerequisites
* **Python 3.10+** (managed via `uv` for lightning-fast environments)
* **Node.js** (v16+)
* **Apple Silicon (M1/M2/M3)** is highly recommended to run the local `laya-mlx` models smoothly.

### 1. Start the FastAPI + Laya Backend
The backend powers the Laya AI Semantic Router. It uses `uvicorn` and FastAPI.

Open a terminal in the root directory:
```bash
# We use `uv` for fast dependency management
uv run uvicorn laya.source.server:app --reload --port 8000
```
*Note: Upon first launch, the `laya-mlx` agent will download the model weights (aac6fef/laya-mlx) automatically.*

### 2. Start the React Frontend Dashboard
The frontend runs the Federated Learning simulation and the UI dashboard.

Open a **new** terminal in the `frontend` directory:
```bash
cd frontend
npm install
npm run dev
```

### 3. Run the Simulation
1. Open your browser to `http://localhost:5173`.
2. You will see the devices in a "Healthy" state.
3. Click the green **Start Simulation** button in the top right.
4. Watch as the Edge devices generate synthetic data, compute local loss, and execute FedAvg weight synchronizations. 
5. When a device enters a Warning or Critical state, you will see the Laya Semantic Router instantly evaluate the payload and trigger the high-contrast Warning/Shutdown banners!

---

## Tech Stack
* **Frontend:** React, Vite, CSS (Glassmorphism UI), Lucide-React Icons. (Includes a custom JS implementation of Logistic Regression + FedAvg).
* **Backend:** Python, FastAPI, Uvicorn, Pydantic.
* **AI Engine:** Laya-MLX (`laya-rl-agent`).
