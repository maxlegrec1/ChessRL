import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
from utils.parse_value_leela import leela_data_generator

class SimpleValueNet(nn.Module):
    """
    A simple convolutional neural network to predict the value of a chess position.
    """
    def __init__(self):
        super(SimpleValueNet, self).__init__()
        # Input shape: (B, 112, 8, 8)
        self.conv1 = nn.Conv2d(112, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        # The generator provides data in shape (B, 1, 8, 8, 112), channels last.
        # We reshape it to what PyTorch Conv2d expects: (B, C, H, W).
        # Squeeze to (B, 8, 8, 112), then permute.
        x = x.squeeze(1).permute(0, 3, 1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train():
    """
    Main training loop.
    """
    config = {
        "learning_rate": 1e-4,
        "total_steps": 50000,
        "leela_config_path": "/mnt/2tb/LeelaDataReader/lczero-training/tf/configs/example.yaml",
        "wandb_project": "ChessRL-pretrain"
    }

    wandb.init(project=config["wandb_project"], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleValueNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Check if config file exists
    if not os.path.exists(config["leela_config_path"]):
        print(f"Leela config file not found at: {config['leela_config_path']}")
        return
    try:
        data_generator = leela_data_generator(config_path=config["leela_config_path"])
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("Starting training...")
    model.train()
    for step in range(config["total_steps"]):

        pos, _, _, value, _ = next(data_generator)
        
        pos = pos.to(device)
        value = value.to(device)

        optimizer.zero_grad()
        
        # The forward pass will handle the reshape
        predictions = model(pos)

        loss = criterion(predictions, value)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted_labels = torch.max(predictions, 1)
        _, true_labels = torch.max(value, 1)
        correct_predictions = (predicted_labels == true_labels).sum().item()
        batch_size = pos.size(0)
        accuracy = correct_predictions / batch_size
        
        if step % 100 == 0:
            print(f"Step {step}/{config['total_steps']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        wandb.log({"value_loss": loss.item(), "value_accuracy": accuracy})



    print("Training finished.")

if __name__ == "__main__":
    train() 