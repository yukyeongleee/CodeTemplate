import torch
from lib import checkpoint, utils
from lib.faceswap import FaceSwapInterface
from simswap.nets import Generator_Adain_Upsample
from submodel.discriminator import Discriminator
from simswap.loss import SimSwapLoss


class SimSwap(FaceSwapInterface):
    def __init__(self, args, gpu):
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        super().__init__(args, gpu)

    def initialize_models(self):
        self.G = Generator_Adain_Upsample().cuda(self.gpu).train()
        self.D = Discriminator().cuda(self.gpu).train()

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.gpu]).module
        
    def load_checkpoint(self, step=-1):
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(0, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(0, 0.999))

    def set_loss_collector(self):
        self._loss_collector = SimSwapLoss(self.args)

    def train_step(self):
        I_source, I_target, same_person = self.load_next_batch()

        ###########
        # Train G #
        ###########

        # Run G to swap identity from source to target image
        I_swapped = self.G(I_source, I_target)
        I_cycle = self.G(I_target, I_swapped)

        id_source = self.G.get_id(I_source)
        id_swapped = self.G.get_id(I_swapped)

        g_real = self.D(I_target)
        g_fake = self.D(I_swapped.detach())
        
        G_dict = {
            "I_source": I_source,
            "I_target": I_target, 
            "I_swapped": I_swapped,
            "I_cycle": I_cycle,

            "same_person": same_person,

            "id_source": id_source,
            "id_swapped": id_swapped,

            "g_real": g_real,
            "g_fake": g_fake
        }

        loss_G = self.loss_collector.get_loss_G(G_dict)
        utils.update_net(self.opt_G, loss_G)

        ###########
        # Train D #
        ###########

        d_real = self.D(I_target)
        d_fake = self.D(I_swapped.detach())

        D_dict = {
            "d_real": d_real,
            "d_fake": d_fake,
        }

        loss_D = self.loss_collector.get_loss_D(D_dict)
        utils.update_net(self.opt_D, loss_D)

        return [I_source, I_target, I_swapped]

    def save_checkpoint(self, step):
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=step)

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)
        utils.save_image(self.args, step, "valid_imgs", [self.valid_source, self.valid_target, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)
        
    @property
    def loss_collector(self):
        return self._loss_collector
