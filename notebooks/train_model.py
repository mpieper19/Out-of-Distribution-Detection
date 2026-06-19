import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DenseNet121(nn.Module):
    
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def train_one_epoch(model, training_loader, loss_fn, optimizer, device):

    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            print(f'  batch {i + 1} avg loss: {running_loss / (i+1)}')

    return running_loss / len(training_loader)


def train_model(model, num_epochs, training_loader, validation_loader, device, model_name):
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    best_vloss = 1_000_000.

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, training_loader, loss_fn, optimizer, device)
        train_losses.append(avg_loss)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        val_losses.append(avg_vloss)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        scheduler.step(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'models/best_model_{model_name}.pt'
            torch.save(model.state_dict(), model_path)

    # Plot train + val loss

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()