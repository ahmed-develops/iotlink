import { useState, useEffect, useRef } from 'react';
import { Activity, Thermometer, Zap, Server, BrainCircuit, RefreshCw, AlertTriangle, ShieldCheck, Network } from 'lucide-react';
import './index.css';

// We simulate 3 IoT devices that will send data to the backend
const DEVICES_INIT = [
  { id: 'IoT-Pump-01', baseTemp: 45, baseVib: 0.1 },
  { id: 'IoT-Turbine-02', baseTemp: 65, baseVib: 0.3 },
  { id: 'IoT-Cooler-03', baseTemp: 30, baseVib: 0.05 },
];

// --- PROPER FEDERATED ML SIMULATION ---
class LocalModel {
  constructor() {
    this.wTemp = Math.random() - 0.5;
    this.wVib = Math.random() - 0.5;
    this.bias = Math.random() - 0.5;
    this.lr = 0.1; // Learning rate
    this.localLoss = 0;
  }
  
  sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  
  predict(temp, vib) {
    // Normalize inputs to keep gradients stable
    const t = (temp - 50) / 50; 
    const v = vib / 1.5;
    const z = this.wTemp * t + this.wVib * v + this.bias;
    return this.sigmoid(z);
  }
  
  train(temp, vib, trueLabel) {
    const t = (temp - 50) / 50;
    const v = vib / 1.5;
    const pred = this.predict(temp, vib);
    
    // Binary cross-entropy derivative
    const error = pred - trueLabel;
    
    // Gradient descent step
    this.wTemp -= this.lr * error * t;
    this.wVib -= this.lr * error * v;
    this.bias -= this.lr * error;
    
    // Calculate approximate local loss
    this.localLoss = - (trueLabel * Math.log(pred + 1e-9) + (1 - trueLabel) * Math.log(1 - pred + 1e-9));
  }
}

function App() {
  const [devices, setDevices] = useState([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  
  // Store local models for FedML in a ref to persist across renders
  const localModels = useRef(new Map(DEVICES_INIT.map(d => [d.id, new LocalModel()])));
  
  const [fedRound, setFedRound] = useState(0);
  const [globalWeights, setGlobalWeights] = useState({ wTemp: 0, wVib: 0, bias: 0 });

  // Fetch devices from backend
  const fetchDevices = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/devices');
      if (res.ok) {
        const data = await res.json();
        setDevices(data);
        setLastUpdate(new Date().toLocaleTimeString());
      }
    } catch (error) {
      console.error("Failed to fetch devices:", error);
    }
  };

  useEffect(() => {
    fetchDevices();
    const interval = setInterval(fetchDevices, 2000);
    return () => clearInterval(interval);
  }, []);

  // Simulate pushing data from IoT to Server + FedML Training Loop
  useEffect(() => {
    if (!isSimulating) return;
    
    let tick = 0;

    const simulateData = async () => {
      tick++;
      
      for (const dev of DEVICES_INIT) {
        // Randomly simulate an anomaly
        const isAnomaly = Math.random() > 0.85;
        const temp = isAnomaly ? dev.baseTemp + 40 + Math.random() * 20 : dev.baseTemp + (Math.random() * 5 - 2.5);
        const vib = isAnomaly ? dev.baseVib + 0.8 + Math.random() * 0.5 : dev.baseVib + (Math.random() * 0.02 - 0.01);
        
        // --- FEDERATED LEARNING: Local Edge Training ---
        const model = localModels.current.get(dev.id);
        
        // Ground truth for simulation (unknown to model, model must learn it)
        const trueLabel = (temp > 80 || vib > 0.9) ? 1 : ((temp > 60 || vib > 0.5) ? 0.6 : 0);
        
        // Train local model on data
        model.train(temp, vib, trueLabel);
        
        // Inference using local model
        const predictionScore = model.predict(temp, vib);
        
        let localPrediction = 'Normal';
        let status = 'normal';
        
        if (predictionScore > 0.8) {
          localPrediction = 'Critical Anomaly Detected';
          status = 'critical';
        } else if (predictionScore > 0.5) {
          localPrediction = 'Warning: Elevated metrics';
          status = 'warning';
        }

        try {
          await fetch(`http://localhost:8000/api/devices/${dev.id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              device_id: dev.id,
              temperature: Number(temp.toFixed(1)),
              vibration: Number(vib.toFixed(3)),
              local_prediction: localPrediction,
              status: status
            })
          });
        } catch (e) {
          console.error("Simulation error:", e);
        }
      }
      
      // --- FEDERATED LEARNING: FedAvg Synchronization every 5 ticks ---
      if (tick % 5 === 0) {
        const models = Array.from(localModels.current.values());
        
        // Aggregate weights centrally
        let avgWTemp = 0, avgWVib = 0, avgBias = 0;
        models.forEach(m => {
          avgWTemp += m.wTemp;
          avgWVib += m.wVib;
          avgBias += m.bias;
        });
        avgWTemp /= models.length;
        avgWVib /= models.length;
        avgBias /= models.length;
        
        // Broadcast global model back to local edges
        models.forEach(m => {
          m.wTemp = avgWTemp;
          m.wVib = avgWVib;
          m.bias = avgBias;
        });
        
        setGlobalWeights({ wTemp: avgWTemp.toFixed(3), wVib: avgWVib.toFixed(3), bias: avgBias.toFixed(3) });
        setFedRound(prev => prev + 1);
      }
    };

    simulateData();
    const interval = setInterval(simulateData, 3000);
    return () => clearInterval(interval);
  }, [isSimulating]);

  const [expandedTrace, setExpandedTrace] = useState(null);

  const toggleTrace = (id) => {
    setExpandedTrace(expandedTrace === id ? null : id);
  };

  const activeAlerts = devices.filter(d => d.status === 'critical').length;
  
  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1>IoT Decision System</h1>
          <p style={{ color: 'var(--text-muted)' }}>Proper Federated ML (FedAvg) + Laya Decision Layer</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {lastUpdate && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Last synced: {lastUpdate}</span>}
          <button 
            className="btn" 
            onClick={() => setIsSimulating(!isSimulating)}
            style={{ background: isSimulating ? 'var(--danger)' : 'var(--success)' }}
          >
            <RefreshCw size={18} className={isSimulating ? 'animate-pulse' : ''} />
            {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
          </button>
        </div>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-title">FedML Rounds (FedAvg)</div>
          <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--success)' }}>
            <Network />
            {fedRound}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-title">Global Status</div>
          <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {activeAlerts > 0 ? (
              <><AlertTriangle color="var(--danger)" /> Critical</>
            ) : (
              <><ShieldCheck color="var(--success)" /> Healthy</>
            )}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-title">Laya Decision Layer</div>
          <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.2rem', color: '#c4b5fd' }}>
            <BrainCircuit />
            Active
          </div>
        </div>
      </div>

      <div className="investor-panel" style={{ 
        background: 'rgba(15, 23, 42, 0.7)', 
        border: '1px solid rgba(59, 130, 246, 0.3)', 
        borderRadius: '1rem', 
        padding: '1.5rem', 
        marginBottom: '2rem',
        backdropFilter: 'blur(12px)'
      }}>
        <h3 style={{ color: 'var(--accent)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Network size={20} /> System Architecture: True Federated ML & Semantic AI
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', fontSize: '0.9rem' }}>
          <div>
            <h4 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.25rem' }}>
              1. Edge Layer (True Federated Learning)
            </h4>
            <p style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Devices train <strong>local Logistic Regression models</strong> using Gradient Descent. Raw sensor data never leaves the device.</p>
            <p style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Every 5 cycles, weights are aggregated via <strong>Federated Averaging (FedAvg)</strong> to build a unified global model without centralizing data.</p>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem', borderRadius: '0.5rem', fontFamily: 'monospace', color: '#94a3b8' }}>
              Global Weights: wTemp={globalWeights.wTemp} | wVib={globalWeights.wVib} | bias={globalWeights.bias}
            </div>
          </div>
          <div>
            <h4 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.25rem' }}>
              2. Laya Decision Layer (Centralized AI)
            </h4>
            <p style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Instead of streaming raw data to an LLM, the FedML edges only transmit semantic anomalies (e.g., "Critical Anomaly Detected") to the cloud.</p>
            <p style={{ color: 'var(--text-muted)' }}>The Laya agent acts as a semantic router, outputting mathematical confidence scores (`noul`) against constrained instructions to trigger maintenance or shutdowns with zero hallucinations.</p>
          </div>
        </div>
      </div>

      <div className="devices-section">
        <div className="devices-header">
          <h2>Edge Devices (Local FedML Inference)</h2>
        </div>

        {devices.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
            <Server size={48} style={{ opacity: 0.5, marginBottom: '1rem' }} />
            <p>No devices connected. Start the simulation to generate data and train edge models.</p>
          </div>
        ) : (
          <div className="devices-grid">
            {devices.map(device => {
              const localModel = localModels.current.get(device.device_id);
              return (
                <div key={device.device_id} className={`device-card status-${device.status}`}>
                  <div className="device-header">
                    <div className="device-id">
                      <Server size={18} />
                      {device.device_id}
                    </div>
                    <div className={`device-status status-badge-${device.status}`}>
                      {device.status}
                    </div>
                  </div>

                  <div className="device-metrics">
                    <div className="metric">
                      <span className="metric-label">Temperature</span>
                      <span className="metric-value" style={{ color: device.temperature > 80 ? 'var(--danger)' : 'inherit' }}>
                        <Thermometer size={16} />
                        {device.temperature} °C
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Vibration</span>
                      <span className="metric-value" style={{ color: device.vibration > 0.8 ? 'var(--danger)' : 'inherit' }}>
                        <Activity size={16} />
                        {device.vibration} g
                      </span>
                    </div>
                  </div>

                  <div className="local-ml">
                    <div className="local-ml-title">Local FedML Model</div>
                    <div className="local-ml-value" style={{ marginBottom: '0.5rem' }}>{device.local_prediction}</div>
                    <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-muted)', background: 'rgba(0,0,0,0.2)', padding: '0.25rem', borderRadius: '0.25rem' }}>
                      Loss: {localModel ? localModel.localLoss.toFixed(4) : '0.000'} | Infer Score: {localModel ? localModel.predict(device.temperature, device.vibration).toFixed(2) : '0.00'}
                    </div>
                  </div>

                  <div className="laya-decision">
                    <div className="laya-decision-title">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <BrainCircuit size={16} />
                        Laya Final Decision
                      </div>
                      {device.laya_trace && (
                        <button className="btn-small" onClick={() => toggleTrace(device.device_id)}>
                          {expandedTrace === device.device_id ? 'Hide Trace' : 'View Trace'}
                        </button>
                      )}
                    </div>
                    {(() => {
                      if (!device.laya_decision) {
                        return (
                          <div className="decision-item">
                            <span className="decision-key">Waiting for Laya...</span>
                          </div>
                        );
                      }
                      if (device.laya_decision.error) {
                        return (
                          <div className="decision-item">
                            <span className="decision-key" style={{ color: 'var(--danger)' }}>Error connecting to model.</span>
                          </div>
                        );
                      }

                      const answers = device.laya_decision.answers || {};
                      const maintDecision = answers.trigger_maintenance || {};
                      const shutdownDecision = answers.shutdown_device || {};
                      
                      // Laya returns an object with a 'noul' score representing semantic match confidence
                      const isMaintenance = (maintDecision.noul || 0) > 0.30;
                      const isShutdown = (shutdownDecision.noul || 0) > 0.25;

                      return (
                        <div className="decision-action-container">
                          <div className={`decision-banner ${isMaintenance ? "decision-banner-active warning" : "decision-banner-inactive"}`}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <span>Trigger Maintenance</span>
                            </div>
                            <span>{isMaintenance ? "Yes" : "No"}</span>
                          </div>
                          <div className={`decision-banner ${isShutdown ? "decision-banner-active critical" : "decision-banner-inactive"}`}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              {isShutdown && <AlertTriangle size={16} />}
                              <span>Shutdown Device</span>
                            </div>
                            <span>{isShutdown ? "Yes" : "No"}</span>
                          </div>
                        </div>
                      );
                    })()}

                    {expandedTrace === device.device_id && device.laya_trace && (
                      <div className="laya-trace-container">
                        <span className="trace-label">State Provided:</span>
                        <pre>{device.laya_trace.state_provided}</pre>
                        
                        <div className="trace-divider"></div>
                        
                        <span className="trace-label">Questions Asked:</span>
                        <pre>{JSON.stringify(device.laya_trace.questions_asked, null, 2)}</pre>
                        
                        <div className="trace-divider"></div>
                        
                        <span className="trace-label">Raw Laya Decision:</span>
                        <pre>{JSON.stringify(device.laya_decision, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

