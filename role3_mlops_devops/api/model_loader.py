"""
Model loading utility.
Loads the model artifact (model.pkl or PyTorch checkpoint) at startup.
Designed so that swapping the file triggers a reload without code changes.
"""

# TODO: Implement load_model(path) with pickle / torch.load support
