import sys
sys.path.append("./")
sys.path.append("../")

import yaml
from lib.config import Config

# with open("your_model/configs.yaml") as f:
#     configs = yaml.load(f, Loader=yaml.FullLoader)

args = Config.from_yaml("your_model/configs.yaml")
print(args)
args.a = 1
print(args)