def get_encoder():
    import sys
    sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
    print("loading model")
    from AE.autoencoder import Autoencoder
    ae = Autoencoder.load_from_checkpoint(
        "codes/AE/lightning_logs/(lay,hid)=(5,64)_sigmoid/checkpoints/epoch=187-step=15792.ckpt",
        inputDim=6,
        compressDim=2,
        layers=5,
        type="rec",
        lr=1e-3,
        hiddenDim=64
    )
    return ae
def get_encode_range(data,range_=range(-3,3,10)):
    import numpy as np
    from torch import Tensor
    encodes = []
    ae = get_encoder()
    for r in range_:
        data[-1]=r
        encodes.append(ae(Tensor(data)).detach().numpy())
    return np.array(encodes)
def get_ori_data():
    import sys
    sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
    from ENCODE.encode import get_encoded,tensor2points
    print("loading points and labels")
    import numpy as np
    labels = np.load("codes/ENCODE/cluster/second/labels/73_(1.20e-02,120).npy")
    return {"points":tensor2points(get_encoded()),"labels":labels}
if __name__ == "__main__":
    from read_incomplete import get_uni_inc
    import numpy as np
    ye4tai4range = np.linspace(-3,3,100)
    data = get_uni_inc().values[0]
    unk = get_encode_range(data)
    knw = get_ori_data()
    import matplotlib.pyplot as plt
    labels = np.load("codes/ENCODE/cluster/second/labels/73_(1.20e-02,120).npy")
    plt.scatter(*knw.T,s=0.2,alpha=0.5,c=labels,cmap="jet")
    plt.scatter(*unk.T,s=10,marker="+",c="k")
    plt.show()