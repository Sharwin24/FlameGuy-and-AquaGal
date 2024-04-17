import torch.nn as nn

# initialzing convolutional neural network
class Net(nn.Module):
    def __init__(self, action_space_size):
        super(Net, self).__init__()
        
        # initiating shared layers
        self.shared_layers = nn.Sequential(
            # convolutional layers
            nn.Conv2d(1, 32, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # flattening
            nn.Flatten(),
            
            # linear layers
            nn.LazyLinear(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        # initializing actor layers
        self.policy_layers = nn.Sequential(
            nn.LazyLinear(action_space_size)
        )
        
        # initializing critic layers
        self.value_layers = nn.Sequential(
            nn.LazyLinear(1)
        )
        
    # get model output for value
    def value(self, x):
        shared = self.shared_layers(x)
        value = self.value_layers(shared)
        return value
    
    # get model output for policy logits
    def policy(self, x):
        shared = self.shared_layers(x)
        policy_logits = self.policy_layers(shared)
        return policy_logits
    
    # perform both tasks
    def forward(self, x):
        shared = self.shared_layers(x)
        policy_logits = self.policy_layers(shared)
        value = self.value_layers(shared)
        return policy_logits, value