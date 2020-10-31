from models.gwcnet_init import GwcNet_init
from models.gwcnet import GwcNet_G, GwcNet_GC
from models.loss import model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-init": GwcNet_init
}
