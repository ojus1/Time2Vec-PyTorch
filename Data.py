from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ToyDataset(Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
        
        df = pd.read_csv("./data/toy_dataset.csv")
        self.x = df["x"].values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

if __name__ == "__main__":
    dataset = ToyDataset()
    print(dataset[6])