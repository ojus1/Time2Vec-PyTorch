from Data import ToyDataset
from periodic_activations import SineActivation, CosineActivation
import torch
from torch.utils.data import DataLoader
from Pipeline import AbstractPipelineClass
from torch import nn
from Model import Model

class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
    
    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset()
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        num_epochs = 100

        for ep in range(num_epochs):
            for x, y in dataloader:
                optimizer.zero_grad()

                y_pred = self.model(x.unsqueeze(1).float())
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()
                
                print("epoch: {}, loss:{}".format(ep, loss.item()))
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x

if __name__ == "__main__":
    pipe = ToyPipeline(Model("sin", 42))
    pipe.train()

    #pipe = ToyPipeline(Model("cos", 12))
    #pipe.train()
