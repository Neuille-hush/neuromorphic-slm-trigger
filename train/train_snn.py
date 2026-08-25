import torch
import torch.nn as nn
import snntorch as snn
import json

class SpikeGateSNN(nn.Module):
    def __init__(self, num_inputs=1, num_hidden=8, num_outputs=2, beta=0.8, threshold=1.0):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

def export_weights(model, filename="config/snn_weights.json"):
    data = {
        "num_inputs": model.fc1.in_features,
        "num_hidden": model.fc1.out_features,
        "num_outputs": model.fc2.out_features,
        "lif1_beta": float(model.lif1.beta),
        "lif1_threshold": float(model.lif1.threshold),
        "lif2_beta": float(model.lif2.beta),
        "lif2_threshold": float(model.lif2.threshold),
        "fc1_weight": model.fc1.weight.detach().tolist(),
        "fc1_bias": model.fc1.bias.detach().tolist(),
        "fc2_weight": model.fc2.weight.detach().tolist(),
        "fc2_bias": model.fc2.bias.detach().tolist(),
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Exported SNN configuration to {filename}")

if __name__ == "__main__":
    model = SpikeGateSNN()
    export_weights(model)
