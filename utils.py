import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def allot_device(random_seed_value):
    if torch.cuda.is_available():
        device = "cuda"
        torch.manual_seed(random_seed_value)
    else:
        device = "cpu"
    return device


class Load_dataset:
    def __init__(self, BATCH_SIZE, shuffle=True, transforms=None):
        self.BATCH_SIZE = BATCH_SIZE
        self.shuffle = True
        self.transforms = transforms

    def get_dataset(self):
        if self.transforms == None:
            train_transforms = transforms.Compose([
                transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                transforms.Resize((28, 28)),
                transforms.RandomRotation((-15., 15.), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transforms, test_transforms = self.transforms[0], self.transforms[1]

        train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
        test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

        kwargs = {'batch_size': self.BATCH_SIZE, 'shuffle': self.shuffle, 'num_workers': 4, 'pin_memory': True}

        train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

        return train_loader, test_loader


class Plots:
    def __init__(self, num_images=None, loaded_data=None, metrics=None):
        self.num_images = num_images
        self.loaded_data = loaded_data
        self.metrics = metrics

    def plot_images(self):
        batch_data, batch_label = next(iter(self.loaded_data))
        fig = plt.figure()
        if self.num_images % 2 != 0:
            self.num_images -= 1
        self.num_rows = self.num_images // 4

        fig = plt.figure(figsize=(15, 7))
        counter = 0
        for i in range(self.num_images):
            sub = fig.add_subplot(self.num_rows, 4, i + 1)
            im2display = (np.squeeze(batch_data[i].permute(2, 1, 0)))
            sub.imshow(im2display.cpu().numpy())
            sub.set_title(batch_label[i].item())
            sub.axis('off')

        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def plot_metrics(self):
        if self.metrics is None:
            print("Please provide the metric values, unable to view them!")
        else:
            train_losses, train_acc, test_losses, test_acc = self.metrics
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0, 0].plot(train_losses)
            axs[0, 0].set_title("Training Loss")
            axs[1, 0].plot(train_acc)
            axs[1, 0].set_title("Training Accuracy")
            axs[0, 1].plot(test_losses)
            axs[0, 1].set_title("Test Loss")
            axs[1, 1].plot(test_acc)
            axs[1, 1].set_title("Test Accuracy")

train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

class Performance:

    def __init__(self, device, model, data,optimizer,criterion):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.test_loader = data[0], data[1]

    def GetCorrectPredCount(self,pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        train_acc.append(100 * correct / processed)
        train_losses.append(train_loss / len(self.train_loader))

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss

                correct += self.GetCorrectPredCount(output, target)

        test_loss /= len(self.test_loader.dataset)
        test_acc.append(100. * correct / len(self.test_loader.dataset))
        test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

def scores():
    return train_losses,train_acc,test_losses,test_acc