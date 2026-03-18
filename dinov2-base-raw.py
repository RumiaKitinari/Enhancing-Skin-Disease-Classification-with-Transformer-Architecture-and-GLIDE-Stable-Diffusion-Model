model_checkpoint = "facebook/dinov2-base" # pre-trained model from which to fine-tune
batch_size = 32 # batch size for training and evaluation
from huggingface_hub import notebook_login

notebook_login()