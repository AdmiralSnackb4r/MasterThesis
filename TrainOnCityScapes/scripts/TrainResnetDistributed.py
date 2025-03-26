import os
import time
import torch
import argparse
import torch.nn as nn
import torch.utils.data.distributed
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import torchvision.transforms.v2 as v2
from distributed_utils import *
from torch.utils.data import DataLoader
from CustomCocoDataset import CustomCocoDataset as coco
from torch.utils.tensorboard import SummaryWriter




def setup_model(resnet_x):
    if resnet_x == 50:
        model = models.resnet50(num_classes=27)
        model.fc = nn.Linear(model.fc.in_features, 27)
    return model

def send_batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch

def train_model(model, train_loader, criterion, optimizer, device):
    """
        Train the model on the entire training dataset and return the global loss.
    """
    model.train()
    total_loss = 0
    #print("begin loading")
    #before_load = time.perf_counter()
    for batch in train_loader:


        batch = send_batch_to_device(batch, device)
        #after_load = time.perf_counter()

        #print(f"Time to load a batch {after_load - before_load}")
        output = model(batch['image'])

        loss = criterion(output, batch['label'])
        loss.backward()
        total_loss += loss

        optimizer.step()
        optimizer.zero_grad()
        before_load = time.perf_counter()
        
    
    result = total_loss / len(train_loader)
    if args.distributed:
        torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
    return result

def test_model(model, val_loader, criterion, device):
    """
        Evaluate the model on an evaluation set and return the global
        loss over the entire evaluation set.
    """
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch in val_loader:

            batch = send_batch_to_device(batch, device)
            output = model(batch['image'])

            loss += criterion(output, batch['label'])
            

    result = loss / len(val_loader)
    if args.distributed:
        torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
    return result


def main(args):

    if args.on_cluster:
        writer = SummaryWriter(log_dir="/p/home/jusers/kromm3/jureca/master/TrainOnCityScapes/logs")
    else:
        writer = SummaryWriter(log_dir="TrainOnCityScapes\\logs")

    transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.distributed:
        local_rank, rank, device = setup()
        print0(local_rank, device)
    else:
        device = "cuda"
    model = setup_model(args.resnet)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,device_ids=[local_rank])

    if args.on_cluster:
        
        train_dataset = coco(args.datapath, annotation_file="/p/project/hai_1008/kromm3/TrainOnCityScapes/CityScapes/annotations/train_dataset.json",
                            mode="object", transforms=transform)
        val_dataset = coco(args.datapath, annotation_file="/p/project/hai_1008/kromm3/TrainOnCityScapes/CityScapes/annotations/valid_dataset.json",
                            mode="object", transforms=transform)
        test_dataset = coco(args.datapath, annotation_file="/p/project/hai_1008/kromm3/TrainOnCityScapes/CityScapes/annotations/test_dataset.json",
                            mode="object", transforms=transform)
    else:
        train_dataset = coco(args.datapath, annotation_file="TrainOnCityScapes\\CityScapes\\annotations\\train_dataset.json",
                            mode="object", transforms=transform)
        val_dataset = coco(args.datapath, annotation_file="TrainOnCityScapes\\CityScapes\\annotations\\valid_dataset.json",
                            mode="object", transforms=transform)
        test_dataset = coco(args.datapath, annotation_file="TrainOnCityScapes\\CityScapes\\annotations\\test_dataset.json",
                            mode="object", transforms=transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=int(0), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    start_time = time.perf_counter()

    for epoch in range(args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = test_model(model, val_loader, criterion, device)

        writer.add_scalar('Loss/Training', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        if args.distributed:
            print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}')
        else:
            print(f"[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # We allow only rank=0 to save the model
            if args.distributed:
                save0(model.state_dict(), 'model_best.pth')
            else:
                torch.save(model.state_dict(), 'model_best.pth')


        scheduler.step()
        

    end_time = time.perf_counter()
    if args.distributed:
        print0('Finished training after', end_time - start_time, 'seconds.')
    else:
        print(f"Finished training after {end_time - start_time} seconds")

    time.sleep(10)

    if args.on_cluster:
        model.load_state_dict(torch.load('/p/project/hai_1008/kromm3/model_best.pth', map_location=device))
    else:
        model.load_state_dict(torch.load('model_best.pth', map_location=device))


    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank]
    # )
    
    test_loss = test_model(model, test_loader, criterion, device)
    writer.add_scalar("Loss/Test", test_loss.item())
    if args.distributed:
        print0('Final test loss:', test_loss.item())
    else:
        print(f"Final test loss: {test_loss.item()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed ResNet Training")
    parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=.002, help='learning rate')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resnet', type=int, default=50, help='version of resnet to train 18/34/50/101/152')
    #parser.add_argument('--datapath', default="/p/scratch/hai_1008/kromm3/CityScapes/leftImg8bit", help='path to data')
    parser.add_argument('--datapath', default="S:\\Datasets\\CityScapes\\leftImg8bit", help='path to data')
    parser.add_argument('--imagesize', type=int, default=224, help='size of image')
    parser.add_argument('--on_cluster', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
