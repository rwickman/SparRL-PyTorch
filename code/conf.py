import torch

# Number of global statistics
NUM_GLOBAL_STATS = 2

# Number of local statistics
NUM_LOCAL_STATS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")