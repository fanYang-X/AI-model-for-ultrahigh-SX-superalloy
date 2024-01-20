import copy
import time
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10e9
    train_loss = []
    val_loss = []
    save_dict_list = []
    
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch+1,num_epochs))
        print('-' * 10)
        for state in ['train', 'val']:
            if state == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0
            for inputs, labels in dataloaders[state]:
                inputs = [i.to(device) for i in inputs]
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(state == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if state == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs[0].size(0)
            if state == 'val':
                scheduler.step(loss)
            epoch_loss = np.sqrt(running_loss / dataset_sizes[state])
            if state == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(
                state, epoch_loss))
            # deep copy the model
            if state == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        save_dict_list.append(copy.deepcopy(model.state_dict()))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)