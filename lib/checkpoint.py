import torch
import os

        
def load_checkpoint(args, model, optimizer):
    try:
        ckpt_dict = torch.load(args.ckpt_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt_dict['model'], strict=False)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        return ckpt_dict['step']
    except:
        if args.isMaster:
            if args.ckpt_path == "None":
                print("start training from scratch")
            else:
                print(f"Failed to load checkpoint; given path is {args.ckpt_path}")
        return 0


def save_checkpoint(args, model, optimizer, name, global_step):
    ckpt_dict = {}
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()
    ckpt_dict['step'] = global_step

    dir_path = f'./train_result/{args.run_id}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = f'{dir_path}/{name}_{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = f'{dir_path}/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        