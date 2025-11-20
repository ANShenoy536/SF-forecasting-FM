import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_model(model, epochs, learning_rate, criterion, optimizer, train_loader):
    hist = []
    # Setting the model to training mode
    model.train()
    # Training
    # print('Training....')
    for i in range(epochs):
        epoch_loss = []
        for batch, (X_train, y_train) in enumerate(train_loader):
            # perform forward pass
            y_pred = model(X_train)  # shape -> (32,2)

            # compute the loss
            loss = criterion(y_pred, y_train)
        
            # Set the gradients of loss to 0
            optimizer.zero_grad()
 
            # Compute the gradients of the loss w.r.t. the weights
            loss.backward()

            # Update the weights
            optimizer.step()

            # Store the loss of each batch in the history list
            epoch_loss.append(loss.detach().item())

        hist.append(sum(epoch_loss) / len(epoch_loss))
        # Print the loss for each epoch
        print(f'Epoch: {i + 1}  Loss = {sum(epoch_loss) / len(epoch_loss)}')
    return model, hist


def evaluate_model(test_loader, model, criterion, device, disp = True):
    test_loss_bw = [] # batchwise test loss
    y_test_all = torch.tensor([], device = device)
    y_pred_all = torch.tensor([], device = device)

    model.eval()
    with torch.no_grad():
        for te_batch, (X_test, y_test) in enumerate(test_loader):
            # get the model prediction (score)
            y_pred_test = model(X_test)

            # compute the loss
            loss = criterion(y_pred_test, y_test)

            # append this loss to the batch_wise test loss
            test_loss_bw.append(loss.item())

            # append the y_test to the y_test_all
            y_test_all = torch.cat((y_test_all, y_test), dim = 0)

            # append the y_pred to y_pred_all list
            y_pred_all = torch.cat((y_pred_all, y_pred_test), dim = 0)

            #Print some message for each batch
            if disp:
               print(f'Testing batch {te_batch + 1}/{len(test_loader)}')
    return torch.tensor(test_loss_bw, device = device), y_test_all, y_pred_all
