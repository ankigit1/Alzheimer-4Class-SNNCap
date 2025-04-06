import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import SiameseCapsuleNetwork
from utils import load_reference_embeddings, classes, device, transform
import os

def single_image_inference(img_path):
    print("\nRunning Inference...")

    # Load model
    model = SiameseCapsuleNetwork()
    checkpoint = torch.load("checkpoints/checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Load reference embeddings
    reference_embeddings = load_reference_embeddings("embeddings/reference_embeddings.pt")

    # Transform image
    image = Image.open(img_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    # Get test embedding
    test_embedding = model.capsule_net(image)

    # Distance computation
    distances = []
    for cls in range(4):
        cls_dists = []
        for ref_img in reference_embeddings[cls]:
            ref_emb = model.capsule_net(ref_img).squeeze(0)
            dist = F.pairwise_distance(test_embedding, ref_emb.unsqueeze(0))
            cls_dists.append(dist.item())
        distances.append(np.mean(cls_dists))

    # Show prediction and probabilities
    pred_label = int(np.argmin(distances))
    confidence_scores = [((max(distances) - d) / (sum(max(distances) - np.array(distances)))) * 100 for d in distances]

    print(f"Predicted Class: {classes[pred_label]}")
    for i, cls in enumerate(classes):
        print(f"{cls}: {confidence_scores[i]:.2f}%")

    # Show image and bar chart
    img = Image.open(img_path)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title("Test Image")

    plt.subplot(1, 2, 2)
    plt.bar(classes, confidence_scores)
    plt.ylabel("Confidence (%)")
    plt.title("Class Likelihoods")
    plt.tight_layout()
    plt.show()