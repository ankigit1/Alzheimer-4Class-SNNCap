import os
import torch
import torch.nn.functional as F
from utils import save_reference_embeddings, compute_distance_based_validation

def train_model(model, train_loader, val_loader, optimizer, criterion,
                reference_embeddings, original_dataset, transform, device):
    num_epochs = 100
    patience = 10
    best_test_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for img1, img2, targets in train_loader:
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        if epoch + 1 == 5:
            reference_embeddings = save_reference_embeddings(
                model, original_dataset, transform, device, mode="refined"
            )

        # Update embeddings every epoch after 6th
        if epoch + 1 > 5:
            reference_embeddings = save_reference_embeddings(
                model, original_dataset, transform, device, mode="update_only"
            )

        accuracy = compute_distance_based_validation(
            model, val_loader, reference_embeddings, device
        )
        test_loss = 1 - accuracy / 100

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'checkpoints/model_best.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break
