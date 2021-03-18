from terra.io import writer, reader
import torch

@writer(torch.Tensor)
def write_tensor(out, path):
    torch.save(out, path)
    return path


@reader(torch.Tensor)
def read_tensor(path):
    return torch.load(path)


class TerraModule:

    def __terra_write__(self, path):
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)

    @classmethod
    def __terra_read__(cls, path):
        # TODO: make it possible to specify a gpu
        dct = torch.load(path, map_location="cpu")
        model = cls(dct["config"])
        model.load_state_dict(dct["state_dict"])
        return model