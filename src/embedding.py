import pickle
from pathlib import Path

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

    for path in Path("assets/lyrics").glob("*.txt"):
        song = path.stem

        # NOTE: 既に処理済みの歌詞はベクトル化・メタデータ化しない
        if Path(f"assets/vectors/{song}.npy").exists():
            continue

        with open(path, "r") as f:
            lyrics = f.read().splitlines()
            lyrics = [lyric for lyric in lyrics if lyric.strip()]

        vectors = [embed(lyric) for lyric in lyrics]
        vectors = np.stack(vectors)

        np.save(f"assets/vectors/{song}.npy", vectors)

        metadata = [[lyric, song] for lyric in lyrics]

        with open(f"assets/metadata/{song}.pkl", "wb") as f:
            pickle.dump(metadata, f)
