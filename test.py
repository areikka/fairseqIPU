# coding=utf-8

import poptorch
import torch
import pdb

class mymodel(torch.nn.Module):
    def __init__(self) :
        super().__init__()
        self.embed = torch.nn.Embedding(128,384)
        self.pipeline_mapping()

    def pipeline_mapping(self):
        self.embed = poptorch.BeginBlock(
            self.embed, "Embedding", ipu_id=0)

    def forward(self, input_ids):
        res = self.embed(input_ids)
        return res

if __name__ == "__main__":
    input = torch.ones((128), dtype=torch.long)
    mm = mymodel()
    pdb.set_trace()
    res = mm(input)
    print(res)