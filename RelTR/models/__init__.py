from .reltr import build
from .CustomRelTR import custom_build

def build_model(args):
    return build(args)

def custom_build_model(args):
    return custom_build(args)