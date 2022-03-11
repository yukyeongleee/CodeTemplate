import os
import gdown

os.makedirs('ptnn', exist_ok=True)

print('Downloading face parsing model...')
gdown.download('https://drive.google.com/uc?id=18E2f82couyeMgSmr3jsdUOh-vHWza2-4', output='ptnn/79999_iter.pth', quiet=False)

print('Downloading faceshifter model...')
gdown.download('https://drive.google.com/uc?id=1dst0ABERyplybSAi2JsmBgU2SQGSVcjH', output='ptnn/G_latest.pth', quiet=False)

print('Downloading arcface model...')
gdown.download('https://drive.google.com/uc?id=1D7xfVpcqhn2v-CmouOlMVyL6zwGmB_fe', output='ptnn/model_ir_se50.pth', quiet=False)

print('Downloading SwinIR model...')
gdown.download('https://drive.google.com/uc?id=182PPfbNlsI6gmw5TNUGn5-7YSujW64ZU', output='ptnn/SR_large.pth', quiet=False)

print('Downloading deep3d model...')
gdown.download('https://drive.google.com/uc?id=1urzp3mGeFq0-2ZfBagfKC-g0_g5y6_-A', output='ptnn/SR_large.pth', quiet=False)

print('Downloading GFPGAN model...')
gdown.download('https://drive.google.com/uc?id=1AJg-8kAZRakiht5GLZURcWXLXS40TdIT', output='ptnn/SR_large.pth', quiet=False)

print('Downloading lpips model...')
gdown.download('https://drive.google.com/uc?id=1I9kxubJzgQ4VClyMFl6CWTAP9RU0b57w', output='ptnn/SR_large.pth', quiet=False)


print('Done.')
