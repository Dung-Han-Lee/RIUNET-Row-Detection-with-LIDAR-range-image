import time
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# User define module
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data_loader, test_loader, criterion, optimizer):
    model.train()
    writer = SummaryWriter("./runs/")
    start_time = time.time()

    for epoch in range(config.num_epoch):
        avg_loss = 0.0

        for batch_num, (weight_map, feats, labels) in enumerate(data_loader):
            (weight_map, feats, labels) = map(lambda x : x.to(device), (weight_map, feats, labels))

            optimizer.zero_grad()
            outputs = model(feats)
            
            loss = criterion(model, outputs, labels, weight_map)
            loss.backward()
            
            for name, param in model.named_parameters():
                try:
                    writer.add_histogram('grad_' + name, param.grad.data, epoch)
                except:
                    pass
                
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 10 == 9:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/10))
                avg_loss = 0.0   
                
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        val_loss   = test_classify(model, test_loader, criterion, optimizer)
        train_loss = test_classify(model, data_loader, criterion, optimizer)
        print('Train Loss: {:.4f} \tVal Loss: {:.4f}'.format(train_loss, val_loss))
        
        end_time = time.time()
        print('Time: ',end_time - start_time, 's')
        
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)

        torch.save( model.state_dict(), "../weights/"+str(epoch)+".pth") 
        
def test_classify(model, test_loader, criterion, optimizer):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (weight_map, feats, labels) in enumerate(test_loader):
        (weight_map, feats, labels) = map(lambda x : x.to(device), (weight_map, feats, labels))
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(model=model,inputs=outputs, target=labels, weight_map=weight_map)
        
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss)
