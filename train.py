import torch
from torch import nn
from models.custom_model import CustomNet
from eval import validate
from data.dataloader import get_dataloaders  

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        #
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

if __name__ == '__main__': 
    # Load data
    train_loader, val_loader = get_dataloaders()

    # Model Initializzation
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    best_acc = 0
    num_epoch = 10

    # Training loop
    for epoch in range(1, num_epoch + 1): 
        train(epoch, model, train_loader, criterion, optimizer)

        val_accuracy = validate(model, val_loader, criterion)

        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy {best_acc:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')