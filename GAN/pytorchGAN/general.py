import torch
import torchvision
import torch.nn as nn  #all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss Function
import torch.optim as optim #for all optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets #Has standard datasets we can import in a nice way
import torchvision.transforms as transforms #transformations we can perform on our dataset
from torch.utils.data import DataLoader #gives easier dataset management and creates minibatches
from torch.utils.tensorboard import SummaryWriter #to print to tensorboard
from model_utils import (Discriminator, Generator) #import our models 


#HyperParameters
lr = 0.0002
batch_size = 64
image_size = 64 #28 x 28 -->64 X64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)(0.5,)),
    ])

dataset = datasets.MNIST(root ='dataset/', train=True, transform=my_transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch .cuda.is_available() else 'cpu')

#create discriminator and generator
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

#setup optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5,0.999))
optimizerG = optim.Adam(netG.Parameters(), lr=lr, betas=(0.5,0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()
real_label = 1
fake_label=0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f'runs/GAN_MNIST/test_real')
writer_fake = SummaryWriter(f'runs/GAN_MNIST/test_fake')

print("Starting Training.....")

for epochs in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1-D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.zeros(batch_size)*0.1).to(device)

        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        ###Train the Generator: maximize log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epochs}/{num_epochs}] Batch {batch_idx}/{Len(dataLoader)} \
                Loss D: {LossD : .4f}, Loss G: {LossG:.4f} D(x): {D_x:.4f}')


            with torch.no_grad():
                fake = netG(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Mnist Real Images', img_grid_real)
                writer_fake.add_image('Mnist fake Images', img_grid_fake)











