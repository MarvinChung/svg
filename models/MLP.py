import torch.nn as nn
import pfrl

class MLP(nn.Module):,
    def __init__(self):
        super(MLP, self).__init__(FixedCovariance=True)
        self.layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),   
        )
        
        if FixedCovariance:
           self.action_head = pfrl.policies.GaussianHeadWithFixedCovariance(scale=1)
        else:
            self.action_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                                    action_size=64,
                                    var_type=\diagonal\,
                                    var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                                    var_param_init=0,  # log std = 0 => std = 1
                                )
            
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.action_head(x)
        return x
