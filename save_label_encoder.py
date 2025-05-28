from dataset import load_dataset
import pickle

dataset_path = "path/to/compiled_dataset/"

try:
    (_,_,_,_), label_encoder = load_dataset(dataset_path)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("label_encoder.pkl has been saved successfully.")
    print(f"Saved to: label_encoder.pkl")
    print(f"Classes:", label_encoder.classes_)

except Exception as e:
    print("Failed to save label encoder: ", e)