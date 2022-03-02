import torch
import torchvision
import torchvision.transforms as transforms
from randomaug import RandAugment


def get_transforms(dataset, use_randaug=False, net_mane='vgg'):
    transform_train = None
    transform_test = None

    if dataset == 'mnist':
        # transforms.Normalize((0.1307,), (0.3081,))
        t = transforms.Normalize((0.5,), (0.5,))
        transform_train = transforms.Compose([transforms.ToTensor(), t])
        transform_test = transforms.Compose([transforms.ToTensor(), t])

    if dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    if dataset == 'cifar10':
        _size = 384 if net_mane == "vit_timm" else 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        _size = 384 if net_mane == "vit_timm" else 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == 'cinic-10':
        # cinic_directory = '/path/to/cinic/directory'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

    if dataset == 'tiny_imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    # Add RandAugment with N, M(hyperparameter)
    if use_randaug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../../Prune/Data', config=None):
    transform_train, transform_test = get_transforms(dataset, config.aug, config.net)
    trainset, testset = None, None
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root + '/mnist', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.MNIST(root=root + '/mnist', train=False, download=True, transform=transform_test)

    if dataset == 'fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(root=root + '/FashionMNIST', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=root + '/FashionMNIST', train=False, download=True, transform=transform_test)

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=False, download=True,
                                               transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cinic-10':
        trainset = torchvision.datasets.ImageFolder(root + '/cinic-10/trainval', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/cinic-10/test', transform=transform_test)

    if dataset == 'tiny_imagenet':
        num_workers = 16
        trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val', transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader
