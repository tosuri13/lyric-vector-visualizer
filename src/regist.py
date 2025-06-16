import pickle

import numpy as np
from torch import FloatTensor
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    vectors = np.load("assets/vectors.npy", allow_pickle=True)

    with open("assets/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    writer = SummaryWriter("logs")
    writer.add_embedding(
        FloatTensor(vectors),
        metadata=metadata,
        metadata_header=["lyric"],
    )
    writer.close()
