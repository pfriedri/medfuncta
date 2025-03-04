import os
import torch
import sys
sys.path.append(".")

import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from common.utils import set_random_seed
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassF1Score


class NFDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data['modulations'].float(), data['label']


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, mode='rgb'):
        super(ResNet50Classifier, self).__init__()
        self.resnet50 = models.resnet50()
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        if mode == 'grayscale':
            self.resnet50.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                                            stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.resnet50(x)


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes, mode='rgb'):
        super(EfficientNetB0Classifier, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0()
        self.efficientnet_b0.classifier[1] = nn.Linear(self.efficientnet_b0.classifier[1].in_features, num_classes)
        if mode == 'grayscale':
            self.efficientnet_b0.features[0][0] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                                                            stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.efficientnet_b0(x)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        labels = labels.squeeze()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.squeeze()
            out = model(data)
            loss = criterion(out, labels)

            total_loss += loss.item()
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            f1_metric.update(predicted, labels)

    accuracy = 100 * correct / total
    f1_score = f1_metric.compute().item()

    return total_loss / len(val_loader), accuracy, f1_score


class Args:
    def __init__(self):
        self.data_dir = "/path/to/medfuncta/set"
        self.classifier = 'simple'  # simple or resnet or efficientnet
        self.mode = 'grayscale'  # grayscale or tgb
        self.batch_size = 32
        self.input_dim = 2048
        self.num_classes = 2
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.seed = 42


def main(args):
    # Define device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Define Dataset and Dataloader """
    if args.classifier == 'simple':
        train_set = NFDataset(os.path.join(args.data_dir, "train"))
        val_set = NFDataset(os.path.join(args.data_dir, "val"))
        test_set = NFDataset(os.path.join(args.data_dir, "test"))
    elif args.classifier == 'resnet' or args.classifier == 'efficientnet':
        from medmnist import PneumoniaMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = PneumoniaMNIST(split='train', transform=transforms, download='True', size=64)
        val_set = PneumoniaMNIST(split='val', transform=transforms, download='True', size=64)
        test_set = PneumoniaMNIST(split='test', transform=transforms, download='True', size=64)
    else:
        raise NotImplementedError()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    """ Select classification model """
    if args.classifier == 'simple':
        model = SimpleClassifier(args.input_dim, args.num_classes).to(device)
    elif args.classifier == 'resnet':
        model = ResNet50Classifier(args.num_classes, mode=args.mode).to(device)
    elif args.classifier == 'efficientnet':
        model = EfficientNetB0Classifier(args.num_classes, mode=args.mode).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {pytorch_total_params}")

    """ Define optimization criterion and optimizer """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    """ Run training and validation loop """
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, args.num_classes)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch + 1}/{args.num_epochs}: ",
              f"Train Loss: {train_loss:.4f} ",
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    """ Final evaluation on test set """
    model.load_state_dict(best_model)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, args.num_classes)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    args = Args()
    main(args)
