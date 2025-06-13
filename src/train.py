from scripts.Trainer import Trainer
import argparse
import albumentations as A
from torch.utils.data import Dataset
from models.GraphNet import GraphNet
from data.dataloader import DATASET
from utils.helpers import *
from utils.loss import DiceBCELoss
from data.transforms import get_train_transforms
from utils.logger import Logger

size = (384, 384)

def main(args, dict):
    # random seed
    torch.manual_seed(args.seed)
    # use gpu or not
    if args.gpu is True and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    (train_img, train_mask, train_edge), (valid_img, valid_mask, valid_edge) = load_dataset( dict['data_path'])  # path = "datasets/Kvasir-SEG"
    path_list_img, path_list_mask, path_list_edge = shuffling(train_img, train_mask, train_edge)

    # load datasets
    train_dataset = DATASET(path_list_img, path_list_mask, path_list_edge, size, transform=None)
    valid_dataset = DATASET(valid_img, valid_mask, valid_edge, size, transform=None)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.nk,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=args.bs,
                                             num_workers=args.nk,
                                             shuffle=False)

    # create model
    model = GraphNet().to(device)

    # freeze backbone
    if args.fb is True:
        for param in model.encoder.parameters():
            param.requires_grad = False

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params_to_optimize, lr=dict['lr'])

    loss = DiceBCELoss()

    logger = Logger(os.path.join(args.scp, 'log.txt'))
    # Trainer: class, defined in trainer.py
    training = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.bs,
        train_loader=train_loader,
        val_loader=val_loader,
        train_log_path=dict['train_log_path'],
        size=size,
        device=device,
        checkpoint_path=args.scp,
        num_epoch=dict['num_epoch'],
        lr=dict['lr'],
        loss_region=loss,
        logger=logger
    )

    training.train()


if __name__ == '__main__':
    # Set params
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='manual seed')
    parser.add_argument('--bs', default=16, type=int, help='batch size for train')
    parser.add_argument('--gpu', default=True, type=bool, help='train on gpu')
    parser.add_argument('--scp', default='checkpoints/BUSI/', type=str, help='save_checkpoint_path')
    parser.add_argument('--yaml_path', default='configs/train.yaml', type=str, help='training settings')
    parser.add_argument('--nk', default=0, type=int, help='num of workers')
    parser.add_argument('--fb', default=True, type=str, help='Freeze Backbone')

    args = parser.parse_args()
    dict = read_dict(args.yaml_path)

    main(args, dict)
