import numpy as np

def main():
    embeddings_E = np.load("word_embeddings_E.npy")
    word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()
    print("loaded")
    words_to_search = ["pes", "škola", "krásný", "auto", "motor", "motocykl", "stroj", "kolo"]
    for word in words_to_search:
        emb = embeddings_E[word_to_idx[word]]
        # nejbližší slova
        distances = np.linalg.norm(embeddings_E - emb, axis=1)
        closest_indices = np.argsort(distances)[:10]
        closest_words = [list(word_to_idx.keys())[i] for i in closest_indices]
        print(f"word: {word}, closest frobenious words: {closest_words}")

    # closest words based on cosine similarity
    print("cosine:")
    for word in words_to_search:
        emb = embeddings_E[word_to_idx[word]]
        # nejbližší slova
        distances = np.dot(embeddings_E, emb) / (np.linalg.norm(embeddings_E, axis=1) * np.linalg.norm(emb))
        closest_indices = np.argsort(distances)[-10:]
        closest_words = [list(word_to_idx.keys())[i] for i in closest_indices]
        print(f"word: {word}, closest cos words: {closest_words}")


    cmd = ""
    while cmd != "exit":
        cmd = input("Enter a word to find its closest words (or 'exit' to quit): ").lower()
        if cmd in word_to_idx:
            emb = embeddings_E[word_to_idx[cmd]]
            # nejbližší slova
            distances = np.linalg.norm(embeddings_E - emb, axis=1)
            closest_indices = np.argsort(distances)[:10]
            closest_words = [list(word_to_idx.keys())[i] for i in closest_indices]
            print(f"word: {cmd}, closest frobenious words: {closest_words}")
        else:
            print(f"word {cmd} not found in vocabulary")


if __name__ == "__main__":
    main()
