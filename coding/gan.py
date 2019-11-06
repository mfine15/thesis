#%%
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor
from torch.autograd import Variable

import torch

import pandas as pd
import numpy as np
cuda = True if torch.cuda.is_available() else False

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#%%
LEARNING_RATE = 0.01
EPOCHS = 1000
LATENT_DIM = 10
CLIP_VALUE = 0.1
N_CRITIC = 5
#%%

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

dtype = {
    "age": np.int32,
    "workclass": "category",
    "fnlweight": np.int32,
    "education": "category",
    "education-num": np.int32,
    "marital-status": "category",
    "occupation": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": np.int32,
    "capital-loss": np.int32,
    "hours-per-week": np.int32,
    "native-country": "category",
    "income": np.int32
}



columns = dtype.keys()
adult = pd.read_csv("adult/adult.clean",names=columns, dtype=dtype, index_col=False)
df = pd.get_dummies(adult)

tensor = data.TensorDataset(torch.from_numpy(df.to_numpy()))
dataloader = data.DataLoader(tensor, batch_size = 32, shuffle = True)
columns = df.shape[1]
#%%

class Generator(nn.Module):
    def __init__(self, D_out):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, D_out),
        )

        if cuda:
            self.model = self.model.cuda()

    def forward(self, z):
        out = self.model(z)
        return out

class Discriminator(nn.Module):
    def __init__(self, D_in):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(D_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.ReLU(0.2)
        )

        if cuda:
            self.model = self.model.cuda()
    def forward(self, sample):
        return self.model(sample)

G = Generator(columns)
D = Discriminator(columns)

if cuda:
    G.cuda()
    D.cuda()


optimizer_G = torch.optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
optimizer_D = torch.optim.RMSprop(D.parameters(), lr=LEARNING_RATE)


#%% Training
batches_done = 0
iters = 0
for epoch in range(EPOCHS):
    for i, [batch] in enumerate(dataloader):
        iters += 1
        true_samples = Variable(batch.type(Tensor))

        optimizer_D.zero_grad()

        z = Variable(Tensor(np.random.normal(0,1, (len(batch), LATENT_DIM))))
        fake_samples = G(z).detach()

        loss_fn = nn.BCELoss()

        true_target = torch.ones(len(true_samples), dtype=torch.float)
        fake_target = torch.zeros(len(fake_samples), dtype=torch.float)

        # loss_D = torch.mean(torch.log(D(true_samples))) + torch.mean(torch.log(1 - D(fake_samples)))
        loss_D = -torch.mean(D(true_samples)) + torch.mean(D(fake_samples))

        loss_D.backward()
        optimizer_D.step()

        # Clip discriminator weights
        for p in D.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        # Train generator every N_CRITIC iterations
        if i % N_CRITIC:
            optimizer_G.zero_grad()
            fake_samples = G(z)
            loss_G = -torch.mean(D(fake_samples))

            loss_G.backward()
            optimizer_G.step()
            if (i  % 100):

                writer.add_scalar("D_loss", loss_D.item(), iters)
                writer.add_scalar("G_loss", loss_G.item(), iters)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, EPOCHS, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

        batches_done += 1
#%%
G(Tensor(np.random.normal(0,1, (len(batch), LATENT_DIM))))
pd.crosstab(adult.age, adult.education)
