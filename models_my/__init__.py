from .spiking_detr_my import build


def build_model(args):
    return build(args)
