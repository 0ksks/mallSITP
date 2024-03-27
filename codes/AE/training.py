from torch.utils.data import DataLoader
from pytorch_lightning import Trainer,loggers
from typing import Literal
import sys
sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
from AE.autoencoder import Autoencoder
def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3, persistent_workers=True)
def train_unit(layers,max_epochs,type:Literal["tri","rec"],hiddenDim=None,lr=1e-3):
    model = Autoencoder(6,2,layers,type,lr,hiddenDim)
    save_dir = "code/AE"
    trainer = Trainer(max_epochs=max_epochs,  log_every_n_steps=50, logger=loggers.TensorBoardLogger(save_dir=save_dir,version=f"(lay,hid)=({layers},{hiddenDim})_sigmoid"))
    trainer.fit(model, dataloader)
def get_trained_model()->Autoencoder:
    print("loading model")
    ae = Autoencoder.load_from_checkpoint(
        "codes/AE/lightning_logs/(lay,hid)=(5,64)_sigmoid/checkpoints/epoch=187-step=15792.ckpt",
        inputDim=6,
        compressDim=2,
        layers=5,
        type="rec",
        lr=1e-3,
        hiddenDim=64
    )
    ae.eval()
    return ae
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
