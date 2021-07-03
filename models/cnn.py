import torch.nn as nn
import pfrl

class CNN(nn.Module):,
    def __init__(self):
        super(CNN, self).__init__(FixedCovariance=False)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2)
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2)
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.linear(?,action_size)
        )
        
        if FixedCovariance:
           self.action_head = pfrl.policies.GaussianHeadWithFixedCovariance(scale=1)
        else:
            self.action_head = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                                    action_size=128,
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
