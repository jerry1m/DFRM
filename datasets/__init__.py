import torch
import copy
from .isbi2016 import ISBI2016_CLASS_NAMES, ISBI2016Dataset
from .mvtec import MVTEC_CLASS_NAMES, MVTecDataset, MVTecFSCopyPasteDataset, MVTecFSDataset, MVTecPseudoDataset, MVTecAnomalyDataset
from .btad import BTAD_CLASS_NAMES, BTADDataset, BTADFSDataset, BTADFSCopyPasteDataset
from .visa import VISA_CLASS_NAMES, VisADataset, VisAFSDataset, VisAFSCopyPasteDataset
from .utils import BalancedBatchSampler


def _resolve_test_dataset_name(args):
    return (getattr(args, 'test_dataset', None) or args.dataset)


def _resolve_test_class_name(args, test_dataset_name):
    test_class_name = getattr(args, 'test_class_name', None)
    if test_class_name:
        return test_class_name
    if test_dataset_name == 'isbi2016':
        return 'lesion'
    return args.class_name


def _clone_dataset_args(args, dataset_name, class_name=None, data_path=None):
    cloned = copy.copy(args)
    cloned.dataset = dataset_name
    if class_name is not None:
        cloned.class_name = class_name
    if data_path is not None:
        cloned.data_path = data_path
    return cloned

__all__ = ['MVTEC_CLASS_NAMES',
           'MVTecDataset',
           'MVTecFSCopyPasteDataset',
           'MVTecFSDataset',
           'MVTecPseudoDataset',
           'MVTecAnomalyDataset',
           'BTAD_CLASS_NAMES',
           'BTADDataset',
           'BTADFSDataset',
           'BTADFSCopyPasteDataset',
           'ISBI2016_CLASS_NAMES',
           'ISBI2016Dataset',
           'VISA_CLASS_NAMES',
           'VisADataset',
           'VisAFSDataset',
           'VisAFSCopyPasteDataset',
           'create_data_loader',
           'create_fas_data_loader',
           'create_test_data_loader']


def create_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    test_dataset_name = _resolve_test_dataset_name(args)
    test_class_name = _resolve_test_class_name(args, test_dataset_name)
    if args.dataset == 'mvtec':
        train_dataset = MVTecDataset(args, is_train=True)
        test_dataset  = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'mvtec' else None
    elif args.dataset == 'btad':
        train_dataset = BTADDataset(args, is_train=True)
        test_dataset  = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'btad' else None
    elif args.dataset == 'visa':
        train_dataset = VisADataset(args, is_train=True)
        test_dataset  = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'visa' else None
    elif args.dataset == 'isbi2016':
        train_dataset = ISBI2016Dataset(args, is_train=True)
        test_dataset  = ISBI2016Dataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'isbi2016' else None
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    if test_dataset is None:
        if test_dataset_name == 'mvtec':
            test_dataset = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'btad':
            test_dataset = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'visa':
            test_dataset = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'isbi2016':
            test_dataset = ISBI2016Dataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        else:
            raise NotImplementedError('{} is not supported test dataset!'.format(test_dataset_name))
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return train_loader, test_loader


def create_fas_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    test_dataset_name = _resolve_test_dataset_name(args)
    test_class_name = _resolve_test_class_name(args, test_dataset_name)
    if args.dataset == 'mvtec':
        normal_dataset = MVTecDataset(args, is_train=True)
        if args.data_strategy == '0':
            train_dataset = MVTecFSDataset(args, is_train=True)
        elif args.data_strategy == '0,1':
            train_dataset = MVTecFSCopyPasteDataset(args, is_train=True)
        elif args.data_strategy == '0,2':
            train_dataset = MVTecPseudoDataset(args, is_train=True)
        elif args.data_strategy == '0,1,2':
            train_dataset = MVTecAnomalyDataset(args, is_train=True)
        if args.not_in_test:
            # Collect images used as anomalies in training to exclude from test set
            excluded_images = []
            # Direct attribute on simple datasets
            if hasattr(train_dataset, 'a_imgs'):
                excluded_images = list(getattr(train_dataset, 'a_imgs'))
            else:
                # Composite dataset (e.g., MVTecAnomalyDataset)
                if hasattr(train_dataset, 'copy_paste_dataset') and hasattr(train_dataset.copy_paste_dataset, 'a_imgs'):
                    excluded_images.extend(list(train_dataset.copy_paste_dataset.a_imgs))
                if hasattr(train_dataset, 'pseudo_dataset') and hasattr(train_dataset.pseudo_dataset, 'a_imgs'):
                    excluded_images.extend(list(train_dataset.pseudo_dataset.a_imgs))
            test_dataset  = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False, excluded_images=excluded_images if len(excluded_images) > 0 else None) if test_dataset_name == 'mvtec' else None
        else:
            test_dataset  = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'mvtec' else None
    elif args.dataset == 'btad':
        normal_dataset = BTADDataset(args, is_train=True)
        if args.data_strategy == '0':
            train_dataset = BTADFSDataset(args, is_train=True)
        elif args.data_strategy == '0,1':
            train_dataset = BTADFSCopyPasteDataset(args, is_train=True)
        if args.not_in_test:
            # Best-effort exclusion for BTAD variants
            excluded_images = []
            if hasattr(train_dataset, 'a_imgs'):
                excluded_images = list(getattr(train_dataset, 'a_imgs'))
            test_dataset  = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False, excluded_images=excluded_images if len(excluded_images) > 0 else None) if test_dataset_name == 'btad' else None
        else:
            test_dataset  = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'btad' else None
    elif args.dataset == 'visa':
        normal_dataset = VisADataset(args, is_train=True)
        if args.data_strategy == '0':
            train_dataset = VisAFSDataset(args, is_train=True)
        elif args.data_strategy in ['0,1', '0,2', '0,1,2']:
            train_dataset = VisAFSCopyPasteDataset(args, is_train=True)
        else:
            raise NotImplementedError('Unsupported data_strategy {} for visa'.format(args.data_strategy))
        if args.not_in_test:
            excluded_images = []
            if hasattr(train_dataset, 'a_imgs'):
                excluded_images = list(getattr(train_dataset, 'a_imgs'))
            test_dataset  = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False, excluded_images=excluded_images if len(excluded_images) > 0 else None) if test_dataset_name == 'visa' else None
        else:
            test_dataset  = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'visa' else None
    elif args.dataset == 'isbi2016':
        normal_dataset = ISBI2016Dataset(args, is_train=True)
        train_dataset = ISBI2016Dataset(args, is_train=True)
        test_dataset = ISBI2016Dataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False) if test_dataset_name == 'isbi2016' else None
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    if test_dataset is None:
        if test_dataset_name == 'mvtec':
            test_dataset = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'btad':
            test_dataset = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'visa':
            test_dataset = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        elif test_dataset_name == 'isbi2016':
            test_dataset = ISBI2016Dataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
        else:
            raise NotImplementedError('{} is not supported test dataset!'.format(test_dataset_name))
    # dataloader
    normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    if args.dataset == 'isbi2016':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    elif args.balanced_data_loader:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(args, train_dataset), **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return normal_loader, train_loader, test_loader


def create_test_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    test_dataset_name = _resolve_test_dataset_name(args)
    test_class_name = _resolve_test_class_name(args, test_dataset_name)
    if test_dataset_name == 'mvtec':
        test_dataset  = MVTecDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
    elif test_dataset_name == 'btad':
        test_dataset  = BTADDataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
    elif test_dataset_name == 'visa':
        test_dataset  = VisADataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
    elif test_dataset_name == 'isbi2016':
        test_dataset  = ISBI2016Dataset(_clone_dataset_args(args, test_dataset_name, class_name=test_class_name, data_path=getattr(args, 'test_data_path', None) or args.data_path), is_train=False)
    else:
        raise NotImplementedError('{} is not supported test dataset!'.format(test_dataset_name))
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return test_loader