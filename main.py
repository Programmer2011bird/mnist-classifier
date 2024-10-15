import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
# import torchvision
import dataloader
import train_test
import torch


EPOCHS: int = 3
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

dataloaders = dataloader.get_data()
train_dataloader = dataloaders[0]
test_dataloader = dataloaders[1]
class_to_idx = dataloaders[2]

def train_main():
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_test.train_step(optimizer, loss_fn, model, train_dataloader)
        test_loss = train_test.test_step(loss_fn, model, train_dataloader)
    
        print(train_loss)
        print(test_loss)

    torch.save(model.state_dict(), "model.pth")

def eval_model(dataloader, model: nn.Module):
    img, label = next(iter(dataloader))
    
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    with torch.inference_mode():
        accuracy = 0

        pred = model(img)
        formatted_preds = torch.argmax(pred, dim=1)
        accuracy += (formatted_preds == label).sum().item()  

        print(formatted_preds)
        print(label)
        print(len(dataloader.dataset))
        print((accuracy / len(dataloader.dataset)) * 10000)
        
        for i in range(len(dataloader)):
            mat_img = img[i].permute(1, 2, 0)
            plt.imshow(mat_img, cmap="grey")
            plt.title(f"True label: {label[i]} | Preds: {formatted_preds[i]}")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    ## uncomment to train the model
    # train_main()
    ## uncomment to see the results of the model
    eval_model(test_dataloader, model)

