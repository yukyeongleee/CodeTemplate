import torch
import wandb
import os
import sys
import numpy as np
from lib.options import BaseOptions

sys.path.append("./")
sys.path.append("./submodel/")
sys.path.append("./submodel/stylegan2")


def train(gpu, args): 

    if args.model_id == 'simswap':
        from simswap.model import SimSwap
        from simswap.options import TrainOptions
        args = TrainOptions().parse()
        model = SimSwap(args, gpu)

    elif args.model_id == 'faceshifter':
        from faceshifter.model import FaceShifter
        from faceshifter.options import TrainOptions
        args = TrainOptions().parse()
        model = FaceShifter(args, gpu)

    elif args.model_id == 'hififace':
        from hififace.model import HifiFace
        from hififace.options import TrainOptions
        args = TrainOptions().parse()
        model = HifiFace(args, gpu)

    elif args.model_id == 'stylerig':
        from stylerig.model import StyleRig
        from stylerig.options import TrainOptions
        args = TrainOptions().parse()
        model = StyleRig(args, gpu)
        
    else:
        print(f"{args.model} is not supported.")
        exit()

    torch.cuda.set_device(gpu)
    args.isMaster = gpu == 0
    model.RandomGenerator = np.random.RandomState(42)
    model.initialize_models()
    model.set_dataset()

    if args.use_mGPU:
        model.set_multi_GPU()

    model.set_data_iterator()
    model.set_validation()
    model.set_optimizers()
    step = model.load_checkpoint()
    model.set_loss_collector()

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # Training loop
    global_step = step if step else 0
    while global_step < args.max_step:
        result = model.train_step()

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                if args.use_wandb:
                    wandb.log(model.loss_collector.loss_dict)
                model.loss_collector.print_loss(global_step)

            # Save image
            if global_step % args.image_cycle == 0:
                model.save_image(result, global_step)

                if args.valid_dataset_root:
                    model.validation(global_step) 

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":
    args = BaseOptions().parse()
    os.makedirs(args.save_root, exist_ok=True)

    # Set up multi-GPU training
    if args.use_mGPU:  
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # Set up single GPU training
    else:
        train(args.gpu_id, args)
