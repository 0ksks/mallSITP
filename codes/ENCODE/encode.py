import torch
def get_encoded()->torch.Tensor:
    import sys
    sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
    from AE.autoencoder import Autoencoder
    from AE.datasets import AEDataset
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
    print("loading dataset")
    dataset = AEDataset('norm')
    print("calculating")
    encoded:torch.Tensor = ae(dataset[:])
    return encoded.view((-1,2))
import torch
import numpy as np
def tensor2points(tensor:torch.Tensor)->np.ndarray:
    """
    return points to plot

    Args:
        tensor (torch.Tensor): (batch_size,dim1,dim2)

    Returns:
        array like: points([[point][point]])
    """
    tensor = tensor.view((tensor.shape[0],-1))
    points = tensor.detach().numpy()
    return points
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.scatter(*tensor2points(get_encoded()).T,s=0.2)
    plt.show()
