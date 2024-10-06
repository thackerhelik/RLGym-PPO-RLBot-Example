import torch
from discrete_policy import DiscreteFF
import os

device = torch.device("cpu")
policy = DiscreteFF(89, 21, [2048, 2048, 1024, 1024], device)
print(policy)
cur_dir = os.path.dirname(os.path.realpath(__file__))
policy.load_state_dict(torch.load(os.path.join(cur_dir, "PPO_POLICY.pt"), map_location=device))
print('done')