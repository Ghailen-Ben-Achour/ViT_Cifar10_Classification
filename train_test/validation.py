from torch.nn import CrossEntropyLoss
import torch

def valid(args, model, test_loader, device):
    print("***VALIDATION***")
    print("---------------")
    model.eval()
    correct = 0
    total_loss = 0
    loss_fct = CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            total_loss += loss_fct(logits[0], y)
            
            preds = torch.argmax(logits[0], dim=-1)

            correct += (preds == y).float().sum()

    accuracy = 100 * correct / (step * args.batch_size)
    total_loss = total_loss / step  
    return accuracy, total_loss
