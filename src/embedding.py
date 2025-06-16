import pickle

import numpy as np
from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI()

    def embed(input: str) -> list[float]:
        response = client.embeddings.create(
            input=input,
            model="text-embedding-3-large",
        )
        embedding = response.data[0].embedding

        print(f"[INFO] Create embedding from text: {input}")

        return embedding

    with open("assets/lyrics/Defying Gravity.txt", "r") as f:
        lyrics = f.read().splitlines()
        lyrics = [lyric for lyric in lyrics if lyric.strip()]

    vectors = [embed(lyric) for lyric in lyrics]
    vectors = np.stack(vectors)

    np.save("assets/vectors.npy", vectors)

    metadata = [[lyric] for lyric in lyrics]

    with open("assets/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
