from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
opt = parser.parse_args()

# 学習後のネットワークを保存するフォルダ作成
try:
    os.makedirs('./models')
except OSError:
    pass
# 生成画像を保存するフォルダ作成
try:
    os.makedirs('./generated_images')
except OSError:
    pass

# 乱数のシード設定
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# GPUの設定
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    ngpu = int(opt.ngpu)
    cudnn.benchmark = True

# データセットの設定
dataset = dset.MNIST(root='./', download=True,
                    transform=transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),]))
# 入力画像のチャネル数
nc=1
# バッチサイズ
batchsize = 64
# Dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)

# 画像を生成するネットワークG
class Generator(nn.Module):
    def __init__(self, ngpu, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nf = 64
        self.main = nn.Sequential(
            # ランダムノイズをDeconvolution layerに入力
            nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.ReLU(),
            # (nf*8) x 4 x 4
            nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(),
            # (nf*4) x 8 x 8
            nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(),
            # (nf*2) x 16 x 16
            nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(),
            # (nf) x 32 x 32
            nn.ConvTranspose2d(self.nf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64の画像を出力
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# 本物と偽物を見分けるネットワークD
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nf = 64
        self.main = nn.Sequential(
            # input size is (nc) x 64 x 64
            nn.Conv2d(nc, self.nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (nf) x 32 x 32
            nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (nf*2) x 16 x 16
            nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (nf*4) x 8 x 8
            nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (nf*8) x 4 x 4
            nn.Conv2d(self.nf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


#ネットワークGに入力するランダムノイズ
nz = 100
fixed_noise = torch.randn(batchsize, nz, 1, 1, device=device)

# ネットワークGの宣言
netG = Generator(opt.ngpu, nz).to(device)

# ネットワークDの宣言
netD = Discriminator(opt.ngpu).to(device)

# 損失関数
criterion = nn.BCELoss()

# ラベルの定義（本物:1, 偽物:0）
real_label = 1
fake_label = 0

# Optimizerの設定
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 画像生成の開始
epoch_num = 20
for epoch in range(epoch_num):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) ネットワークDの学習（＝log(D(x)) + log(1 - D(G(z)))の最大化）
        ###########################
        # 本物の画像の見分ける
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_cpu)
        # 損失(loss)の計算
        errD_real = criterion(output, label)
        # 誤差逆伝搬
        errD_real.backward()
        D_x = output.mean().item()

        # 偽物の画像を見分ける
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        # 損失(loss)の計算
        errD_fake = criterion(output, label)
        # 誤差逆伝播
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        # パラメータ更新
        optimizerD.step()

        ############################
        # (2) ネットワークGの学習（＝log(D(G(z)))の最大化）
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        # 損失(loss)の計算
        errG = criterion(output, label)
        # 誤差逆伝播
        errG.backward()
        D_G_z2 = output.mean().item()
        # パラメータ更新
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epoch_num, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # 生成画像の保存
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), './generated_images/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
    # ネットワークの保存
    torch.save(netG.state_dict(), './models/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), './models/netD_epoch_%d.pth' % (epoch))
