from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS

@HOOKS.register_module()
class UnfreezeDecoderHook(Hook):
    def __init__(self, unfreeze_epoch=3):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            
            for name, param in model.named_parameters():
                if 'decoder' in name:
                    param.requires_grad = True