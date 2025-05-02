import logging
from ..resnet import resnet50

from ...attention.attention import add_attention


logger = logging.getLogger(__name__)


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()
    #特征提取
    model = resnet50(model_params)
    #加入注意力机制
    model = add_attention(args, model)

    return model
