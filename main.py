import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision
import dataloader
import train_test
import torch


EPOCHS: int = 1
LEARNING_RATE: float = 0.01

class mnist_classifier(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape) -> None:
        super().__init__()

        self.Layer1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=in_shape, out_features=hidden_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=out_shape)
        )

    def forward(self, x):
        return self.Layer1(x)

torch.manual_seed(42)
model = mnist_classifier(28*28, 10, 10)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

dataloaders = dataloader.get_data()

train_dataloader = dataloaders[0]
test_dataloader = dataloaders[1]
class_to_idx = dataloaders[2]


for epoch in tqdm(range(EPOCHS)):
    train_loss = train_test.train_step(optimizer, loss_fn, model, train_dataloader)
    test_loss = train_test.test_step(loss_fn, model, train_dataloader)

    print(train_loss)
    print(test_loss)


for X, y in train_dataloader:
    print(y)
    print(class_to_idx)
    pred = model(X)

    print(torch.argmax(pred, dim=1))

    break
