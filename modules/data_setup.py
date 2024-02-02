
import os
import torch 
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

def create_dataloaders(train_path:str,
                      test_path:str,
                      train_transform:torchvision.transforms,
                      test_transform:torchvision.transforms,
                      batch_size:int,
                      num_workers:int):
  """
  Creates train and test dataloaders.

  Keyword Arguments:
    :arg train_path: train directory path
    :type train_path: str or Path 
    :arg test_path: test directory path 
    :type test_path: str or Path 
    :arg train_transform: transform for train data 
    :type train_transform: torch.transforms
    :arg test_transform: transform for test data 
    :type test_transform: torch.transforms
    :arg batch_size: BATCH_SIZE
    :type batch_size: int 
    :arg num_workers: number of cpus 
    :type num_workers: int 

  Example Usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_path=train_path,
                                                                        test_path=test_path,
                                                                        train_transform=train_transform,
                                                                        test_transform=test_transform,
                                                                        batch_size=BATCH_SIZE,
                                                                        num_workers=NUM_WORKERS)

  """
  if not train_transform:
    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])
  if not test_transform:
    test_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])

  train_folder = ImageFolder(root=train_path,
                             transform=train_transform)

  test_folder = ImageFolder(root=test_path,
                            transform=test_transform)

  BATCH_SIZE=128
  NUM_WORKERS=os.cpu_count()

  train_dataloader = torch.utils.data.DataLoader(dataset=train_folder,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 generator=torch.Generator(device="cpu")
                                                )

  test_dataloader = torch.utils.data.DataLoader(dataset=test_folder,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                               shuffle=False,
                                                pin_memory=True,
                                                generator=torch.Generator(device="cpu"))

  class_names = train_folder.classes

  return train_dataloader, test_dataloader, class_names
