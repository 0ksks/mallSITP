print("importing nn,optim,pl")
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Literal
class Autoencoder(pl.LightningModule):
    def __init__(self, inputDim:int, compressDim:int, layers:int, type:Literal["tri","rec"]="tri", lr:float=0.001, hiddenDim:int=None) -> None:
        super(Autoencoder, self).__init__()
        self.lr = lr
        if type == "tri":
            encoder = []
            decoder = []
            encodeDims = np.linspace(inputDim,compressDim,layers,dtype=int)
            decodeDims = np.linspace(compressDim,inputDim,layers,dtype=int)
            for i in range(layers-2):
                encoder.append(nn.Linear(encodeDims[i],encodeDims[i+1]))
                encoder.append(nn.ReLU())
                decoder.append(nn.Linear(decodeDims[i],decodeDims[i+1]))
                decoder.append(nn.ReLU())
            encoder.append(nn.Linear(encodeDims[-2],encodeDims[-1]))
            encoder.append(nn.ReLU())
            decoder.append(nn.Linear(decodeDims[-2],decodeDims[-1]))
        elif type == "rec":
            if hiddenDim is None:
                hiddenDim = inputDim
            encoder = [nn.Linear(inputDim,hiddenDim),nn.ReLU()]
            encoder += [nn.Linear(hiddenDim,hiddenDim),nn.ReLU()]*(layers-2)
            encoder += [nn.Linear(hiddenDim,compressDim),nn.ReLU()]
            decoder = [nn.Linear(compressDim,hiddenDim),nn.ReLU()]
            decoder += [nn.Linear(hiddenDim,hiddenDim),nn.ReLU()]*(layers-2)
            decoder += [nn.Linear(hiddenDim,inputDim)]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
    def forward(self,input):
        encoded = self.encoder(input)
        return encoded
    def training_step(self, batch, batch_idx):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        loss = F.mse_loss(batch, decoded)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
if __name__=="__main__":
    inputTensor = torch.randn((3,1,10))
    model = Autoencoder(10,2,6,'tri')
    outputTensor = model(inputTensor)
    print(f"{inputTensor = }")
    print(f"{outputTensor = }")
    print(model)