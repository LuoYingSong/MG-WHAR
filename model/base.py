from torch import nn
from torch.nn import functional as F

from util.util import makedir

class BaseModel(nn.Module):
    def __init__(self, dataset_name):
        super(BaseModel, self).__init__()
        self.dataset_name = dataset_name
        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    @property
    def path_checkpoints(self):
        return f"data/model/{self.dataset_name}/{self.__class__.__name__}/checkpoints/"

    @property
    def path_logs(self):
        return f"data/model/{self.dataset_name}/{self.__class__.__name__}/logs/"

    @property
    def path_visuals(self):
        return f"data/model/{self.dataset_name}/{self.__class__.__name__}/visuals/"