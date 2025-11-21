# mnist_mlp.py
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from clearml import Task


# ===== モデル定義 (MLP) =====
class MNISTMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="mnist_mlp")
    return parser.parse_args()


def main():
    args = parse_args()

    # ===== ClearML Task 初期化 =====
    task = Task.init(
        project_name="demo-clearml-mnist",
        task_name="mnist_mlp",
        task_type=Task.TaskTypes.training,
    )
    # argparseのハイパーパラメータをUIに出す
    # task.connect(args, name="hyperparameters")
    logger = task.get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== データセット (MNIST) =====
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNISTの平均・分散
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2
    )

    # ===== モデル・損失・オプティマイザ =====
    model = MNISTMLP(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # ===== 学習ループ =====
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
                # ClearMLにtrain lossを送る
                logger.report_scalar(
                    title="loss",
                    series="train",
                    value=avg_loss,
                    iteration=global_step,
                )
                running_loss = 0.0

        # ===== エポックごとにテスト =====
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_test_loss = test_loss / total
        acc = correct / total

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Test Loss: {avg_test_loss:.4f} Acc: {acc:.4f}"
        )

        # ClearMLにval loss / accuracyを送る
        logger.report_scalar(
            title="loss",
            series="val",
            value=avg_test_loss,
            iteration=epoch,
        )
        logger.report_scalar(
            title="accuracy",
            series="val",
            value=acc,
            iteration=epoch,
        )

    # ===== モデル保存 & ClearMLへアップロード =====
    save_path = f"{args.model_name}.pt"
    torch.save(model.state_dict(), save_path)
    task.upload_artifact(name="model_weights", artifact_object=save_path)

    task.close()


if __name__ == "__main__":
    main()
