import argparse
import torch

class BaseOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.gpu_num = torch.cuda.device_count()
        self.initialize()
        
    def initialize(self):

        self.parser.add_argument('--gpu_num', default=self.gpu_num)
        self.parser.add_argument('--run_id', type=str, required=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--isMaster', default=True)

    def parse(self):
        args = self.parser.parse_args()
        return args
