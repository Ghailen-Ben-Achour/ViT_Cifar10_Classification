train.py
Who has access
Not shared
System properties
Type
Text
Size
3 KB
Storage used
3 KB
Location
train_test
Owner
me
Modified
3 Nov 2021 by me
Opened
07:50 by me
Created
29 Oct 2021 with Google Drive for desktop
Add a description
Viewers can download
import torch
from torch.nn import CrossEntropyLoss
from load_data import get_loader
from time import perf_counter
import os
import wandb

from train_test.validation import valid
from utils import save_checkpoint, load_checkpoint

#def save_model(args, model_to_save):
  
  #model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
  #torch.save(model_to_save.state_dict(), model_checkpoint)
  #print("Model saved successfully!")

def train(args, model, device):
  train_loader, test_loader = get_loader(args)

  optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=0.9,
                              weight_decay=args.weight_decay)
  
  loss_fct = CrossEntropyLoss()

  try:
    #model.load_state_dict(torch.load(os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)))
    ckpt = load_checkpoint(args)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['ViT'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print("Loading pretrained model")
  except Exception:
    print(' [*] No checkpoint!')
    start_epoch = 0
  # Prepare dataset
  

  wandb.login()
  #Initialize WandB 
  wandb.init(name=args.name, 
           project='ViT',
           notes='Cifar classification', 
           tags=['First Run'],
           entity='ghailen', 
           id='ViT', resume="allow")
        
        
  wandb.watch([model], log='all')
  for epoch in range(start_epoch, args.epochs):
        print("***TRAINING***")
        print("---------------")
        model.train()
        correct = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            t0 = perf_counter()
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model(x)

            loss = loss_fct(logits[0], y)
            
            preds = torch.argmax(logits[0], dim=-1)

            correct += (preds == y).float().sum()
            loss.backward()
            
            optimizer.step()
            print("Epoch: (%3d/%3d) | Step: (%3d/%3d) | Loss: %.2e | Time: %.2e" % 
            (epoch, args.epochs, step, len(train_loader), loss, perf_counter()-t0))
        
        valid_acc, valid_loss = valid(args, model, test_loader, device)
        print("Epoch: (%3d/%3d) | Val_acc: %.2e | Loss: %.2e " % 
            (epoch, args.epochs, valid_acc, valid_loss))
        
        wandb.log({
                "ViT_Validation_Accuracy": valid_acc,
                "ViT_Validation_Loss" : valid_loss})

        save_checkpoint({'epoch': epoch + 1,
                                'ViT': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                 args)
