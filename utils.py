import torch
from torchvision import transforms
from model import ResizeWithPadding

classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    ResizeWithPadding((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_reference_embeddings(path):
    ref = torch.load(path, map_location=device)
    for cls in ref:
        ref[cls] = [img.to(device) for img in ref[cls]]
    return ref
