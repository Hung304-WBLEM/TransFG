import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017
from .autoaugment import AutoAugImageNetPolicy
from features_classification.datasets import Mass_Shape_Dataset, Mass_Margins_Dataset
from features_classification.datasets import Calc_Type_Dataset, Calc_Dist_Dataset
from features_classification import custom_transforms

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset in ['Mass_Shape', 'Mass_Margins', 'Calc_Type', 'Calc_Dist']:
        input_size = args.img_size
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(25, scale=(0.8, 1.2)),
                custom_transforms.IntensityShift((-20, 20)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        root_dir = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data2'

        if args.dataset == 'Mass_Shape':
            mass_shape_dir = 'mass/cls/mass_shape_comb_feats_omit'

            trainset = Mass_Shape_Dataset(os.path.join(root_dir, mass_shape_dir, 'train'),
                                        data_transforms['train'])
            testset = Mass_Shape_Dataset(os.path.join(root_dir, mass_shape_dir, 'val'),
                                        data_transforms['val'])

            classes = Mass_Shape_Dataset.classes
            train_dir = os.path.join(os.path.join(root_dir, mass_shape_dir), 'train')

        elif args.dataset == 'Mass_Margins':
            mass_margins_dir = 'mass/cls/mass_margins_comb_feats_omit'

            trainset = Mass_Margins_Dataset(os.path.join(root_dir, mass_margins_dir, 'train'),
                                            data_transforms['train'])
            testset = Mass_Margins_Dataset(os.path.join(root_dir, mass_margins_dir, 'val'),
                                           data_transforms['val'])

            classes = Mass_Margins_Dataset.classes
            train_dir = os.path.join(os.path.join(root_dir, mass_margins_dir), 'train')

        elif args.dataset == 'Calc_Type':
            calc_type_dir = 'calc/cls/calc_type_comb_feats_omit'

            trainset = Calc_Type_Dataset(os.path.join(root_dir, calc_type_dir, 'train'),
                                         data_transforms['train'])
            testset = Calc_Type_Dataset(os.path.join(root_dir, calc_type_dir, 'val'),
                                        data_transforms['val'])

            classes = Calc_Type_Dataset.classes
            train_dir = os.path.join(os.path.join(root_dir, calc_type_dir), 'train')

        elif args.dataset == 'Calc_Dist':
            calc_dist_dir = 'calc/cls/calc_dist_comb_feats_omit'

            trainset = Calc_Dist_Dataset(os.path.join(root_dir, calc_dist_dir, 'train'),
                                         data_transforms['train'])
            testset = Calc_Dist_Dataset(os.path.join(root_dir, calc_dist_dir, 'val'),
                                        data_transforms['val'])

            classes = Calc_Dist_Dataset.classes
            train_dir = os.path.join(os.path.join(root_dir, calc_dist_dir), 'train')

    elif args.dataset == 'CUB_200_2011':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader, classes, train_dir
