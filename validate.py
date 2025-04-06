import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def validate_model(model, dataset, transform, device):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    reference_embeddings = torch.load('embeddings/reference_embeddings.pt')

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, true_label = dataset[idx]
            image_tensor = transform(image).unsqueeze(0).to(device)
            test_embedding = model.capsule_net(image_tensor)

            distances = []
            for cls in range(4):
                cls_dists = []
                for ref_img in reference_embeddings[cls]:
                    ref_emb = model.capsule_net(ref_img).squeeze(0)
                    dist = F.pairwise_distance(test_embedding, ref_emb.unsqueeze(0), p=2)
                    cls_dists.append(dist.item())
                distances.append(sum(cls_dists)/len(cls_dists))

            pred_label = int(torch.argmin(torch.tensor(distances)))
            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)

    cm = confusion_matrix(all_true_labels, all_pred_labels)
    print(classification_report(all_true_labels, all_pred_labels))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
