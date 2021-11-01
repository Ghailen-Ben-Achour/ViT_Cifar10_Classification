import torch
import os 

# To save the checkpoint 
def save_checkpoint(state, args):
    torch.save(state, os.path.join(args.output_dir, "%s_checkpoint.ckpt" % args.name))


