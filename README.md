# SpikeGate SNN
An ultra-low-power **Spiking Neural Network (SNN)** gatekeeper written in native C++ designed to continuously monitor bio-signal data streams and trigger on-device **Small Language Model (SLM)** inference via `llama.cpp`.
## Architecture Overview
Continuous Input --> [ SpikeGate LIF SNN ] -- (Membrane Potential < Threshold) --> [ Idle / Sleep ]
                           |
                 (Membrane Potential >= Threshold)
                           |
                           v Output Spike
                  [ llama.cpp (Qwen 2.5) ] --> Tactical Response
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
git clone https://github.com/YOUR_USERNAME/spikegate-snn.git
cd spikegate-snn
```
### 2. Build llama.cpp dependencies
```bash
git clone https://github.com/ggerganov/llama.cpp
cmake -B llama.cpp/build -S llama.cpp
cmake --build llama.cpp/build --config Release -j4
```
### 3. Fetch GGUF weights
```bash
wget -O model.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```
### 4. Compile and Run
```bash
clang++ -O3 src/spikegate.cpp -I src/ -o spikegate && ./spikegate
```
## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
    [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
