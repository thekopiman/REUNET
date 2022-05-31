from __future__ import print_function, division
import base
from lib import loaders, modules
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from skimage.metrics import mean_squared_error, structural_similarity
import cv2
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Reunet(base.BaseRM):
    def __init__(self, transform=transforms.ToTensor(), model=None, device='default'):
        '''
        Version: 3.1
        REUNET is a modular version of all the other Reunet Versions. 
        This includes inaccessible area, upscaling to 256 for wireless-insite data, and multitransmitter setup.
        This also includes Area of Operations setup.
        transforms = defaults to transforms.ToTensor()
        model = Add in the model (via torch.load) inside.
        device = cuda/cpu. Defaults to default is which automatically detects.

        '''
        super().__init__()
        self.pixel_base = 231
        self.inputs = None
        self.transform = transform
        self.model = model
        self.inaccessible = None
        self.areaofops = None
        if device == 'default':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cpu':
            print(
                'Ensure that the model being loaded has map_location=torch.device(\'cpu\')')
            self.device = torch.device('cpu')

        self.model.to(self.device)

    def padding(self, adjustment='auto') -> None:
        '''
        As REUNET only take in 256*256 images as input, a shortcut for smaller images is to add padding
        to make it 256*256.
        The padding will automatically adjust for generate_sparse_data() and read_true_data() if adjustment is auto
        However, you may specify which one you would like to pad only.
        adjustment = 'auto', 'true_data', 'sparse_data'
        '''
        if adjustment == 'auto':
            self.true_data_width = self.true_data.shape[0]
            if self.true_data_width < 256:
                self.true_data = np.pad(self.true_data, ((
                    0, 256-self.true_data_width), (0, 256-self.true_data_width)), 'constant', constant_values=(0))
                self.sparsedata = np.pad(self.sparsedata, ((
                    0, 256-self.true_data_width), (0, 256-self.true_data_width)), 'constant', constant_values=(0))

        elif adjustment == 'true_data':
            self.true_data_width = self.true_data.shape[0]
            if self.true_data_width < 256:
                self.true_data = np.pad(self.true_data, ((
                    0, 256-self.true_data_width), (0, 256-self.true_data_width)), 'constant', constant_values=(0))
        elif adjustment == 'sparse_data':
            self.true_data_width = self.sparsedata.shape[0]
            if self.true_data_width < 256:
                self.sparsedata = np.pad(self.sparsedata, ((
                    0, 256-self.true_data_width), (0, 256-self.true_data_width)), 'constant', constant_values=(0))
        else:
            raise ValueError(
                'Make sure that adjustment is either - auto, true_data or sparse_data')

    def unpad(self, adjustment='auto') -> None:
        '''
        As REUNET only take in 256*256 images as input, a shortcut for smaller images is to add padding
        to make it 256*256. Subsequently, the image is unpadded and returns the original image size.
        The unpad will automatically adjust for generate_sparse_data() and read_true_data() if adjustment is auto
        However, you may specify which one you would like to pad only.
        adjustment = 'auto', 'true_data', 'sparse_data'

        In order for unpad to work, padding need to be ran first.
        '''
        if adjustment == 'auto':
            if self.true_data_width < 256:
                self.true_data = self.true_data[:self.true_data_width -
                                                256, :self.true_data_width-256]
                self.sparsedata = self.sparsedata[:self.true_data_width -
                                                  256, :self.true_data_width-256]
        elif adjustment == 'true_data':
            if self.true_data_width < 256:
                self.true_data = self.true_data[:self.true_data_width -
                                                256, :self.true_data_width-256]
        elif adjustment == 'sparse_data':
            if self.true_data_width < 256:
                self.sparsedata = self.sparsedata[:self.true_data_width -
                                                  256, :self.true_data_width-256]
        else:
            raise ValueError(
                'Make sure that adjustment is either - auto, true_data or sparse_data')

    def convert_to_pixel(self, data='radiomapseer') -> None:
        '''
        The REUNET model requires the input to be in greyscale pixel values.

        data = 'radiomapseer' or 'wireless-insite'
        '''
        if data == 'radiomapseer':
            '''
            RadioMapSeer Images are alr in pixels values
            '''
            # self.true_data = np.floor(255*(self.true_data + 47.84)/99.16)
            pass
        else:
            # self.true_data[self.true_data == -250] = 0
            pseudo_true = self.initial_data.copy()
            pseudo_true[pseudo_true == -250] = 0
            self.normalizing_factor = int(np.amax(
                self.true_data) - np.amin(pseudo_true))
            self.truncate = int(np.amin(pseudo_true))
            self.true_data[self.true_data < self.truncate] = self.truncate
            self.true_data = np.floor(
                (self.true_data-self.truncate)*(self.pixel_base / self.normalizing_factor))
            # The building occupancy of nearest neighbout and cubic might not match up perfectly
            self.sparsedata[self.sparsedata < self.truncate] = self.truncate
            self.sparsedata[self.sparsedata == 0] = self.truncate
            self.sparsedata = np.floor(
                (self.sparsedata-self.truncate)*(self.pixel_base/self.normalizing_factor))

    @classmethod
    def convert_to_pathloss(self, matrix):
        '''
        Convert Pixel values to dBm

        matrix = pixel matrix

        output = dBm matrix
        '''
        return matrix*99.16/255 - 124

    @classmethod
    def convert_to_power(self, matrix):
        '''
        Convert dBm to mW

        matrix = dBm matrix

        output = mW matrix
        '''
        return 10**(matrix/10)

    @classmethod
    def convert_to_pathloss_r(self, matrix):
        '''
        Convert dBm values to Pixel

        matrix = dBm matrix

        output = pixel matrix
        '''
        return (matrix+124)*255/99.16

    @classmethod
    def convert_to_power_r(self, matrix):
        '''
        Convert mw to dBm

        matrix = mW matrix

        output = dBm matrix
        '''
        return 10*np.log10(matrix)

    def generate_inputs(self, automate=True, map_folder='REUNETData/png/buildings_complete/', measurements=200, data='wireless-insite', building_signal_strength=-250, inaccessible=False):
        '''
        This method automatically generate sparse data and other parameters necessary to run the class. This is a pre-requisite 
        to run the class smoothly.

        automate = bool -> Run generate building coordinates and generate sparse data | Default: True
        measurements = number of measurements/sensors present | Default: 200
        data = 'wireless-insite' or 'radiomapseer'
        building_signal_strength = base pixel/dBm value present in the raw data. Values below the specified value is deemed to be a building. | Default: -250 
        radiomapseer -> 0
        wireless-insite -> -164

        '''
        if automate:
            self.generate_building_coordinates(
                building_signal_strength=building_signal_strength, map_folder=map_folder)

            if self.areaofops != None:
                for y, x in self.building_coordinates:
                    self.true_data[y, x] = 0
            # print(self.building_coordinates)
            self.generate_sparse_data(
                feature_extraction=True, measurements=measurements, inaccessible=inaccessible)

        # Convert true_data and sparse_data to pixel values
        self.convert_to_pixel(data=data)

        self.building_map = np.zeros((256, 256))
        for y, x in self.building_coordinates:
            self.building_map[y, x] = 1

        # if self.true_data_width < 256:
        #     self.building_map = np.pad(self.building_map, ((
        #         0, 256-self.true_data_width), (0, 256-self.true_data_width)), 'constant', constant_values=(0))

        # self.sparsedata = cv2.resize(self.sparsedata, dsize=(
        #     256, 256), interpolation=cv2.INTER_LANCZOS4)

        self.inputs = np.stack(
            [self.building_map, self.sparsedata], axis=2)

        if self.transform:
            self.inputs = self.transform(self.inputs).type(torch.float32)
            self.true_image = self.transform(
                self.true_data).type(torch.float32)

        return [self.inputs, self.true_image]

    def return_inputs(self):
        '''
        Returns the input necessary for the class to work. 
        Do not use the method unless necessary.
        '''
        return [self.inputs, self.true_image]

    def run(self, data='wireless-insite') -> None:
        '''
        Run the model here!
        '''
        inputs, self.target = DataLoader(
            self.return_inputs(), batch_size=1, shuffle=False, num_workers=0)
        self.inputs = inputs.to(self.device)

        _, pred = self.model(self.inputs)
        self.output = (pred.detach().cpu().numpy()).astype(np.uint8)

        self.output = self.output[0][0]

    def calculate_mse(self) -> float:
        '''
        Calculate the MSE between the output and ground truth
        '''
        return mean_squared_error(self.output, self.true_data)

    def calculate_ssim(self) -> float:
        '''
        Calculate the SSIM between the output and ground truth
        '''
        return structural_similarity(self.output, self.true_data)

    def read_true_data(self, file_path, file_type="wireless-insite", dimension_length=None, interpolate=True, areaofops=None) -> None:
        '''
        Create a matrix from the IMG file/Wireless-insite text file.
        file_path = Path of the file.
        file_type = 'wireless-insite' or 'radiomapseer'
        dimension_length = Specifies the max dimension_length. | Default: None
        interpolate = Use bicubic interpolation to upscale the ground truth for wireless-insite data | Default: True
        areaofops = Area of ops from centre of the map in (length, width)
        '''
        self.areaofops = areaofops
        self.file_path_idx = file_path.split('/')[-1].split('_')[0]
        super().read_true_data(file_path, file_type, dimension_length)

        if self.areaofops != None:
            self.true_data = self.true_data[int(127 - self.areaofops[0]/2): int(
                128 + self.areaofops[0] / 2), int(127 - self.areaofops[1]/2): int(128 + self.areaofops[1]/2)]
            self.true_data = cv2.resize(
                self.true_data, (256, 256), interpolation=cv2.INTER_LINEAR)

        if interpolate and file_type == 'wireless-insite':
            self.initial_data = self.true_data.copy()
            self.true_data = cv2.resize(self.true_data, dsize=(
                256, 256), interpolation=cv2.INTER_CUBIC)
            self.true_data[self.true_data > np.amax(
                self.initial_data)] = np.amax(self.initial_data)
            self.true_data[self.true_data < np.amin(
                self.initial_data)] = np.amin(self.initial_data)

    def generate_building_coordinates(self, true_map_matrix=None, building_signal_strength=-250, equal=False, map_folder='REUNETData/png/buildings_complete/') -> None:
        '''
        Generates the building coordinates

        true_map_matrix = Uses the ground_truth generated in read_true_data() | Default: None
        building_signal_strength = building_signal_strength = base pixel/dBm value present in the raw data. Values below the specified value is deemed to be a building. | Default: -250 
        radiomapseer -> 0
        wireless-insite -> -164
        equal= True -> Values must be equal to the building_signal_strength to be considered a building. 
               False -> Values less than or equal to the building_signal_strength to be considered a building.
               |Default: False
        '''
        if os.path.isfile(f'{map_folder}/{self.file_path_idx}.png'):
            print(
                f'Obtained Building map from {map_folder}/{self.file_path_idx}.png')
            raw_image = cv2.imread(
                f'{map_folder}/{self.file_path_idx}.png', flags=cv2.IMREAD_GRAYSCALE)

            if self.areaofops != None:
                raw_image = raw_image[int(127 - self.areaofops[0]/2): int(128 + self.areaofops[0] /
                                      2), int(127 - self.areaofops[1]/2): int(128 + self.areaofops[1]/2)]
                raw_image = cv2.resize(
                    raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
                raw_image[raw_image > 0] = 255

            self.building_coordinates = np.transpose(np.nonzero(raw_image))
            return self.building_coordinates
        else:
            print(
                f'{map_folder}/{self.file_path_idx}.png cannot be found')
            return super().generate_building_coordinates(true_map_matrix, building_signal_strength, equal)

    def generate_inaccessible_coordinates(self, radius, shape='circle', centre=(128, 128), path=None, strategic_placement="centre") -> None:
        '''
        Generates a list of coordinates where sensors cannot be placed. 
        Ensure that the argument - inaccessible = True
        when using generate_inputs() or generate_sparse_data()
        if you decide to use this method.

        If the path has something, it will take precedence over everything.
        If strategic_placement is "inside" or "outside", centre will be ignored.

        Input: 
        - shape = 'circle' or 'square'
        - centre = (y,x) where y,x is within 0 to 255
        - path = optional. Image file (Greyscale) of the buildings
        - strategic_placement = "centre"/"inside"/"outside" - Automatically adjusts detects the transmitter location and create the inaccessible area with respect to the location.
        "centre" ignore the transmitter location and the area is the centre.
        "inside", the transmitter is inside the inaccessible area
        "outside", the transmitter is outside the inaccessible area

        Output: None
        '''
        coordinates = []
        try:
            if os.path.isfile(path):
                print(f'Obtain Inaccessible Map from {path}')
                self.inaccessible = np.argwhere(cv2.imread(
                    path, flags=cv2.IMREAD_GRAYSCALE))
            else:
                raise TypeError
        except TypeError:
            if strategic_placement == "centre":
                if shape == 'circle':
                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            if (y-centre[0])**2 + (x-centre[1])**2 <= radius**2:
                                coordinates.append((y, x))
                elif shape == 'square':
                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            coordinates.append((y, x))
                else:
                    raise AttributeError("Shape is either circle or square.")
                self.inaccessible = coordinates
                return
            elif strategic_placement == "inside":
                transmitter_coordinates = np.argwhere(
                    self.true_data == np.amax(self.true_data))
                mean_transmitter_coordinate = (
                    int(transmitter_coordinates[:, 0].mean()), int(transmitter_coordinates[:, 1].mean()))
                possible_centre = []
                possible_coordinates = []

                if shape == 'circle':
                    for y in range(mean_transmitter_coordinate[0] - radius, mean_transmitter_coordinate[0] + radius):
                        for x in range(mean_transmitter_coordinate[1] - radius, mean_transmitter_coordinate[1] + radius):
                            if (y-mean_transmitter_coordinate[0])**2 + (x-mean_transmitter_coordinate[1])**2 <= radius**2:
                                possible_centre.append((y, x))
                    centre = random.choice(possible_centre)
                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            if (y-centre[0])**2 + (x-centre[1])**2 <= radius**2:
                                possible_coordinates.append((y, x))

                elif shape == 'square':
                    for y in range(mean_transmitter_coordinate[0] - radius, mean_transmitter_coordinate[0] + radius):
                        for x in range(mean_transmitter_coordinate[1] - radius, mean_transmitter_coordinate[1] + radius):
                            if (y-mean_transmitter_coordinate[0])**2 + (x-mean_transmitter_coordinate[1])**2 <= radius**2:
                                possible_centre.append((y, x))
                    centre = random.choice(possible_centre)
                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            possible_coordinates.append((y, x))
                else:
                    raise AttributeError("Shape is either circle or square.")
                self.inaccessible = possible_coordinates
                return

            elif strategic_placement == "outside":
                transmitter_coordinates = np.argwhere(
                    self.true_data == np.amax(self.true_data))
                mean_transmitter_coordinate = (
                    int(transmitter_coordinates[:, 0].mean()), int(transmitter_coordinates[:, 1].mean()))

                possible_coordinates = []
                possible_centre = np.ones((256, 256))

                if shape == 'circle':
                    for y in range(mean_transmitter_coordinate[0] - radius, mean_transmitter_coordinate[0] + radius):
                        for x in range(mean_transmitter_coordinate[1] - radius, mean_transmitter_coordinate[1] + radius):
                            if (y-mean_transmitter_coordinate[0])**2 + (x-mean_transmitter_coordinate[1])**2 <= radius**2:
                                if y >= 0 and y < 256 and x >= 0 and x < 256:
                                    possible_centre[y, x] = 0

                    possible_centre = np.argwhere(possible_centre)
                    centre = random.choice(possible_centre)

                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            if (y-centre[0])**2 + (x-centre[1])**2 <= radius**2:
                                possible_coordinates.append((y, x))
                elif shape == 'square':
                    for y in range(mean_transmitter_coordinate[0] - radius, mean_transmitter_coordinate[0] + radius):
                        for x in range(mean_transmitter_coordinate[1] - radius, mean_transmitter_coordinate[1] + radius):
                            possible_centre[y, x] = 0

                    possible_centre = np.argwhere(possible_centre)
                    centre = random.choice(possible_centre)

                    for y in range(centre[0] - radius, centre[0] + radius + 1):
                        for x in range(centre[1] - radius, centre[1] + radius + 1):
                            possible_coordinates.append((y, x))
                else:
                    raise AttributeError("Shape is either circle or square.")
                self.inaccessible = possible_coordinates
                return


if __name__ == '__main__':
    pass
