from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")


class RadioUNet_s_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",  # set maps_inds as a parameter to be used as well
                 ind1=0, ind2=0,  # Remember to Set ind1/2 as parameters
                 dir_dataset="RadioMapSeer/",
                 numTx=2,
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=0,  # not necessary
                 fix_samples=0,
                 data_samples=300,
                 num_samples_low=10,
                 num_samples_high=299,
                 n_iterations=10,
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of Transmitter per map. Default = 2. Number of RM with the same Buildings but different Tx locations
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10.
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:

        """
        # Remember to set custom maps inds and ind

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.n_iterations = n_iterations
        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "IRT4":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain = self.dir_dataset+"gain/carsIRT4/"

        elif simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset+"gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset+"gain/carsIRT2/"

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset+"png/buildings_complete/"
        else:
            # a random index will be concatenated in the code
            self.dir_buildings = self.dir_dataset+"png/buildings_missing"
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.fix_samples = fix_samples
        self.data_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high

        self.transform = transform

        # self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx*self.n_iterations

    def __getitem__(self, idx):

        idxn = np.floor(idx/self.n_iterations).astype(int)
        idxr = np.floor(idxn/self.numTx).astype(int)
        idxc = idxn-idxr*self.numTx
        dataset_map_ind = self.maps_inds[idxr+self.ind1]+1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Rx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(
                self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        # Will be normalized later, after random seed is computed from it
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Rx (receiver):
        # print(name2)
        # # quit()
        # img_name_Rx = os.path.join(self.dir_Rx, name2)
        # image_Rx = np.asarray(io.imread(img_name_Rx))/256

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(
                io.imread(img_name_gain)), axis=2)/256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            # IRT2 weight of random average
            w = np.random.uniform(0, self.IRT2maxW)
            image_gain = w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2)/256  \
                + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2)/256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain = image_gain/(1-self.thresh)

        # we use this normalization so all RadioUNet methods can have the same learning rate.
        image_gain = image_gain*256
        # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
        # Important: when evaluating the accuracy, remember to devide the errors by 256!

        # Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width, self.height))
        # Each map has its fixed samples, independent of the transmitter location.
        seed_map = np.sum(image_buildings)
        np.random.seed(seed_map)
        x_samples = np.random.randint(0, 255, size=self.data_samples)
        y_samples = np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples, y_samples] = 1

        # input samples from the sparse gain samples
        input_samples = np.zeros((256, 256))
        if self.fix_samples == 0:
            num_in_samples = np.random.randint(
                self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples = np.floor(self.fix_samples).astype(int)

        data_inds = range(self.data_samples)
        input_inds = np.random.permutation(data_inds)[0:num_in_samples]
        x_samples_in = x_samples[input_inds]
        y_samples_in = y_samples[input_inds]
        input_samples[x_samples_in,
                      y_samples_in] = image_gain[x_samples_in, y_samples_in, 0]

        # input_samples, sparse_samples = self.generate_sparse_data(
        # image_gain, measurements=400)
        # normalize image_buildings, after random seed computed from it as an int
        image_buildings = image_buildings/256

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack(
                [image_buildings, input_samples], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs = np.stack([image_buildings,
                              input_samples, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)

        return [inputs, image_gain]
