import utils
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join

torch.manual_seed(1234)

def get_loader(data_root, batchsize, poison=False, fracdirty=.5, cifar100=False, noaugment=False, nogan=True, cinic=False, tanti=False, svhn=False, surface=False, nworker=1):
  '''return loaders for cifar'''

  ## transforms
  def get_transform(datamean, datastd):
    transform_train = transforms.Compose([
      # transforms.RandomCrop(32, padding=4),
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(datamean, datastd),
    ])
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(datamean, datastd),
    ])
    transform_tanti = transforms.Compose([
      # transforms.RandomCrop(32, padding=6),
      # transforms.Lambda(transforms.functional.hflip), # temporary
      # transforms.RandomRotation(5),
      transforms.ToTensor(),
      transforms.Normalize(datamean, datastd),
    ])
    transform_switchable = transform_test if noaugment else transform_train
    return transform_train, transform_test, transform_switchable, transform_tanti

  ## multiplex between cifar, cinic, and svhn
  if not cinic and not svhn:
    datamean, datastd = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train, transform_test, transform_switchable, transform_tanti = get_transform(datamean, datastd)
    Dataset = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
    testset = Dataset(root=data_root, train=False, download=True, transform=transform_test)
    args_trainset = dict(root=data_root, train=True, download=True)
  elif cinic:
    cinic_root = join(data_root, 'CINIC-10')
    utils.maybe_download(source_url='https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz',
                         filename='CINIC-10', target_directory=cinic_root, filetype='tar')
    datamean, datastd = [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
    transform_train, transform_test, transform_switchable, transform_tanti = get_transform(datamean, datastd)
    Dataset = torchvision.datasets.ImageFolder
    testset = Dataset(cinic_root+'/test', transform=transform_test)
    args_trainset= dict(root=cinic_root+'/train')
  elif svhn:
    datamean, datastd = [0.43768212, 0.44376972, 0.47280444], [0.1200278, 0.12307685, 0.10515254]
    transform_train, transform_test, transform_switchable, transform_tanti = get_transform(datamean, datastd)
    svhn_root = join(data_root, 'SVHN')
    # trainset = torchvision.datasets.SVHN(svhn_root, 'train', transform=transform_test)
    trainset = torchvision.datasets.SVHN(svhn_root, 'train', transform=transform_train, download=True)
    if not surface:
      testset  = torchvision.datasets.SVHN(svhn_root, 'test', transform=transform_test, download=True)
      ganset  = torchvision.datasets.SVHN(svhn_root, 'extra', transform=transform_test, download=True)

  ## dataset objects
  if svhn:
    pass
  elif poison:
    trainset = Dataset(transform=transform_switchable, **args_trainset)
    if tanti: ganset = Dataset(transform=transform_tanti, **args_trainset)
    elif nogan: trainset, ganset = torch.utils.data.random_split(trainset, [25000, 25000])
    # else: ganset = CifarGan(root=data_root, transform=transform_test if nogan else transform_switchable)
    else: ganset = Dataset(root=cinic_root+'/valid', transform=transform_train)
  else:
    trainset = Dataset(transform=transform_switchable, **args_trainset)

  ## dataloader objects
  if not surface: testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=nworker)
  else: testloader = None
  if poison:
    gansize = int(batchsize * fracdirty)
    trainsize = batchsize - gansize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=nworker)
    ganloader = torch.utils.data.DataLoader(ganset, batch_size=gansize, shuffle=True, num_workers=nworker)
  else:
    trainsize = batchsize
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainsize, shuffle=True, num_workers=nworker)
    ganloader = None

  return trainloader, ganloader, testloader


