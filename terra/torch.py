from terra.io import writer, reader
import torch

@writer(torch.Tensor)
def write_tensor(out, path):
    torch.save(out, path)
    return path


@reader(torch.Tensor)
def read_tensor(path):
    return torch.load(path)
