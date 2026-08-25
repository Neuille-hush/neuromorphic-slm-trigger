# SpikeGate SNN
An ultra-low-power **Spiking Neural Network (SNN)** gatekeeper written in native C++ designed to continuously monitor bio-signal data streams and trigger on-device **Small Language Model (SLM)** inference via `llama.cpp`.
## Architecture Overview
Continuous Input ──► [ SpikeGate LIF SNN ] ──(Membrane Potential < Threshold)──► [ Idle / Sleep ]
                           │
                 (Membrane Potential >= Threshold)
                           │
                           ▼ Output Spike
                  [ llama.cpp (Qwen 2.5) ] ──► Tactical Response
## Highlights
- **Low Idle Power:** The SNN filter evaluates continuous temporal data with minimal energy draw while keeping the LLM unloaded.
- **Fast Wake Latency:** Target anomaly detection triggers execution in **~6ms**.
- **On-Device Inference:** Launches a local `Qwen 2.5 0.5B` GGUF model via ARM CPU optimization in Termux/Android.
## Performance Benchmarks

| Metric | Measurement |
| :--- | :--- |
| **SNN Processing Latency** | `< 1 ms` / step |
| **Detection Time** | `6 ms` |
| **Model Size** | Qwen 2.5 0.5B (390 MB) |
| **Prompt Processing Speed** | `50.2 t/s` |
| **Generation Speed** | `18.9 t/s` |

## Quickstart
### 1. Clone repository
```bash
git clone [https://github.com/YOUR_USERNAME/spikegate-snn.git](https://github.com/YOUR_USERNAME/spikegate-snn.git)
cd spikegate-snn
