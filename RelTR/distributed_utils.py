import os
import functools

import torch
from torch.utils.tensorboard import SummaryWriter

def setup():

    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')

    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))

    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])

    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)

    # Different random seed for each process.
    torch.random.manual_seed(1000 + torch.distributed.get_rank())

    return local_rank, rank, device


functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)

def create_writer(log_dir):
    writer = None
    if is_root_process():
        writer = SummaryWriter(log_dir)
    return writer

def add_stats(writer, train_stats, eval_stats, epoch):
    if is_root_process():
        writer.add_scalar('Train/Loss', sum(train_stats['loss']) / len(train_stats['loss']), epoch)
        writer.add_scalar('Train/Class Error', sum(train_stats['class_error']) / len(train_stats['class_error']), epoch)
        writer.add_scalar('Train/BBox Error', sum(train_stats['loss_bbox']) / len(train_stats['loss_bbox']), epoch)

        writer.add_scalar('Val/Loss', sum(eval_stats['loss']) / len(eval_stats['loss']), epoch)
        writer.add_scalar('Val/Class Error', sum(eval_stats['class_error']) / len(eval_stats['class_error']), epoch)
        writer.add_scalar('Val/BBox Error', sum(eval_stats['loss_bbox']) / len(eval_stats['loss_bbox']), epoch)

def add_all_stats(writer, train_stats, eval_stats, epoch):
    if is_root_process():
        writer.add_scalar('Train/Loss', sum(train_stats['loss']) / len(train_stats['loss']), epoch)
        writer.add_scalar('Train/Class Error', sum(train_stats['class_error']) / len(train_stats['class_error']), epoch)
        writer.add_scalar('Train/BBox Error', sum(train_stats['loss_bbox']) / len(train_stats['loss_bbox']), epoch)
        writer.add_scalar('Train/Sub Error', sum(train_stats['sub_error']) / len(train_stats['sub_error']), epoch)
        writer.add_scalar('Train/Obj Error', sum(train_stats['obj_error']) / len(train_stats['obj_error']), epoch)
        writer.add_scalar('Train/Rel Error', sum(train_stats['rel_error']) / len(train_stats['rel_error']), epoch)


        writer.add_scalar('Val/Loss', sum(eval_stats['loss']) / len(eval_stats['loss']), epoch)
        writer.add_scalar('Val/Class Error', sum(eval_stats['class_error']) / len(eval_stats['class_error']), epoch)
        writer.add_scalar('Val/BBox Error', sum(eval_stats['loss_bbox']) / len(eval_stats['loss_bbox']), epoch)
        writer.add_scalar('Val/Sub Error', sum(eval_stats['sub_error']) / len(eval_stats['sub_error']), epoch)
        writer.add_scalar('Val/Obj Error', sum(eval_stats['obj_error']) / len(eval_stats['obj_error']), epoch)
        writer.add_scalar('Val/Rel Error', sum(eval_stats['rel_error']) / len(eval_stats['rel_error']), epoch)



def close_writer(writer):
    if is_root_process():
        writer.close()