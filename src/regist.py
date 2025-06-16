import pickle
from pathlib import Path

import numpy as np
from torch import FloatTensor
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    all_vectors = []
    all_metadata = []

    for vector_path in Path("assets/vectors").glob("*.npy"):
        song_name = vector_path.stem
        metadata_path = Path(f"assets/metadata/{song_name}.pkl")

        vectors = np.load(vector_path)
        all_vectors.append(vectors)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            all_metadata.extend(metadata)

    writer = SummaryWriter("logs")
    writer.add_embedding(
        FloatTensor(np.vstack(all_vectors)),
        metadata=all_metadata,
        metadata_header=["lyric", "song"],
    )
    writer.close()
