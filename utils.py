import torch
import os 

# To save the checkpoint 
def save_checkpoint(state, args):
    torch.save(state, os.path.join(args.output_dir, "%s_checkpoint.ckpt" % args.name))


def load_checkpoint(args, map_location=None):
    ckpt_path = os.path.join(args.output_dir, "%s_checkpoint.ckpt" % args.name)
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt
