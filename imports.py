import torch
import numpy as np
from scipy.linalg import expm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  random_split, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
# import wandb
# from pytorch_lightning.loggers import WandbLogger
import concurrent.futures
import time
from functools import partial
import jax
import jax.numpy as jnp
import jax.dlpack


# [x] TODO FNN for lifting procedure
# [] TODO Aggregate the final results using [x] max, min, conv2d, [x] pooling
# [x] TODO Update the paper
# [] TODO Understand JAX 
# [] TODO Optimize the JAX code - use profiler and evaluate individual functions
# [x] TODO MNIST dataset 90% accuracy
# [x] TODO Create a lecture of GL Crossed modules
# [] TODO Check if reverse feedback of edges mul of the entire image is face element (TALK today)
# [x] TODO Solve the identity issue (p1*p2*p3*p4 should give Id face)[in visuals file]
# [x] TODO Figure out nan values in Visuals
# [x] TODO Fashion MNIST
