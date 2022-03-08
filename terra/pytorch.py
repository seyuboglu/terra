import os
from typing import Any, Dict

import torch

from terra.io import reader, writer


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
        model = cls(config=dct["config"])
        model.load_state_dict(dct["state_dict"])
        return model

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        dirpath = None
        for k, v in checkpoint["callbacks"].items():
            if isinstance(k, str) and k.startswith("ModelCheckpoint"):
                dirpath = v["dirpath"]
                best_model_path = v["best_model_path"]
                break

        if dirpath is None:
            return

        if os.path.exists(best_model_path):
            # this is not the best model, so do not dump
            return

        from terra import Task

        Task.dump(
            {
                "epoch": checkpoint["epoch"],
                "global_step": checkpoint["global_step"],
                "model": self,
                "valid": self.valid_preds.compute(),
            },
            run_dir=dirpath,
            group_name="best_chkpt",
            overwrite=True,
        )
