from torch.utils.data import DataLoader
from pytorch_lightning import Trainer,loggers
from typing import Literal
def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3, persistent_workers=True)
def train_unit(layers,max_epochs,type:Literal["tri","rec"],hiddenDim=None,lr=1e-3):
    from autoencoder import Autoencoder
    model = Autoencoder(6,2,layers,type,lr,hiddenDim)
    save_dir = "code/AE"
    trainer = Trainer(max_epochs=max_epochs,  log_every_n_steps=50, logger=loggers.TensorBoardLogger(save_dir=save_dir,version=f"(lay,hid)=({layers},{hiddenDim})_sigmoid"))
    trainer.fit(model, dataloader)
if __name__ == "__main__":
    from datasets import AEDataset
    batch_size = 1000
    max_epochs = 1000
    dataloader = get_dataloader(AEDataset('norm'),batch_size)
    # best (5,5)
    layers = 5
    hiddenDim = 64
    train_unit(layers,max_epochs,'rec',hiddenDim)
    
    
'''
tensorboard --logdir=code/AE/lightning_logs
'''
