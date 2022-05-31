import numpy as np
import math


class BaseRM():
    def __init__(self):
        '''
        BaseRM is the basic parent class that is used to build/create new map constructions
        algorithm on. This class main purpose is to generate basic information such as 
        generate_building coordinates and read_true_data.
        '''
        self.true_data = None
        self.building_coordinates = None
        self.sparsedata = None
        self.output = None

    def get_output(self) -> np.ndarray:

        # -----------------------------------------------------------------
        # Check Requirements or else raise Exception
        # -----------------------------------------------------------------

        try:
            if self.output == None:
                raise Exception('output is None as you did not run run().')
            else:
                return self.output
        except ValueError:
            return self.output

    def get_sparse_data(self) -> np.ndarray:
        '''
        Returns np.ndarray matrix of sparsedata. 
        Generally used for comparison sake against estimated RM.
        '''
        try:
            if type(self.sparsedata) == np.ndarray:
                return self.sparsedata
            else:
                raise TypeError
        except TypeError:
            raise TypeError(
                'sparsedata is None as you did not run generate_sparse_data().')

    def get_true_data(self) -> np.ndarray:
        '''
        Returns np.ndarray matrix of sparsedata. 
        Generally used for comparison sake against estimated RM.
        '''
        try:
            if type(self.true_data) == np.ndarray:
                return self.true_data
            else:
                raise TypeError
        except TypeError:
            raise TypeError(
                'true_data is None as you did not run read_true_data().')

    def generate_sparse_data(self, feature_extraction=False, measurements=0, inaccessible=False) -> None:
        '''
        If the data you input is ground truth, it is advisable to use this method to pick up 
        random measurement points. Ensure that read_true_data() has run before using this method.
        If feature_extraction is True, ensure that generate_building_coordinates() has run.
        If feature_extraction is True, sparse_data measurement points will NOT be generated on buildings.

        Input: 
        feature_extraction -> bool
        inaccessible -> bool

        '''

        # -----------------------------------------------------------------
        # Check Requirements or else raise Exception
        # -----------------------------------------------------------------
        try:
            if self.true_data == None:
                raise Exception('Run read_true_data() method first.')
        except ValueError:
            pass
        if feature_extraction == True:
            try:
                if self.building_coordinates == None:
                    raise Exception(
                        'Run generate_building_coordinates() method first.')
            except ValueError:
                pass

        # -----------------------------------------------------------------
        # Generate sparse_data using RNG and true_data
        # -----------------------------------------------------------------
        self.sparsedata = np.zeros(
            (self.true_data.shape[0], self.true_data.shape[1]))

        if feature_extraction or inaccessible:
            total_blocking = set()
            if feature_extraction:
                total_blocking = total_blocking.union(
                    set(map(tuple, self.building_coordinates)))
            if inaccessible:
                total_blocking = total_blocking.union(
                    set(map(tuple, self.inaccessible)))

            possible_coordinates = np.array(list(set(map(tuple, np.argwhere(
                self.true_data))) - total_blocking))

            # testing
            self.sussyman = possible_coordinates
            #
            np.random.shuffle(possible_coordinates)
            possible_coordinates = possible_coordinates[:measurements]

        else:
            possible_coordinates = np.argwhere(self.true_data)
            np.random.shuffle(possible_coordinates)
            possible_coordinates = possible_coordinates[:measurements]

        for coordinate in possible_coordinates:
            self.sparsedata[coordinate[0], coordinate[1]
                            ] = self.true_data[coordinate[0], coordinate[1]]

    def generate_building_coordinates(self, true_map_matrix=None, building_signal_strength=-250, equal=False) -> np.ndarray:
        '''
        Input: true_map_matrix -> Full RM Matrix | if true_map_matrix is None and it will use 
        the true_data generated data in read_true_data

        Output: None
        This method will only work on Wireless Insite RM data where building RSSI/PSD is -250.
        If you have the Value of the RSSI/PSD of another simulation data, please feel free to 
        change the default value of building_signal_strength

        '''
        self.building_signal_strength = building_signal_strength
        try:
            if true_map_matrix == None:
                try:
                    if self.true_data != None:
                        true_map_matrix = self.true_data
                except ValueError:
                    true_map_matrix = self.true_data
                else:
                    raise Exception(
                        'You either run read_true_data() or pass a np.ndarray into the argument true_map_matrix')
        except ValueError:
            pass

        if equal:
            self.building_coordinates = np.argwhere(
                true_map_matrix == building_signal_strength)

        else:
            self.building_coordinates = np.argwhere(
                true_map_matrix <= building_signal_strength)

        return self.building_coordinates

    def read_true_data(self, file_path, file_type="wireless-insite", dimension_length=None) -> None:
        '''
        Reads a txt or img file and create a matrix with it's required data.

        Input:
        file_type -> image or wireless-insite
        dimension_length -> int

        output -> None

        *However, if you use this as an external module and you have a matrix prepared in your
        main program do the following:

        self.true_data = <np.ndarray>

        This method main function is just to edit self.true_data.
        '''
        if file_type == "wireless-insite":
            with open(file_path) as r:
                xy_data = []
                for line in r.readlines()[3:]:
                    linelist = line.split(" ")
                    xy_data.append(
                        (int(linelist[1])/2 - 5, int(linelist[2])/2 - 5, float(linelist[5])))

            if dimension_length == None:
                dimension_length = int(math.sqrt(len(xy_data)))

            if dimension_length == 'auto':
                dimension_length = math.floor(
                    int(math.sqrt(len(xy_data)))/self.upscale) * self.upscale

            true = np.zeros((dimension_length, dimension_length))

            for value in xy_data:
                if not int(value[0]) >= dimension_length and not int(value[1]) >= dimension_length:
                    true[int(value[0]), int(value[1])] = value[2]

        elif file_type == "image":
            import cv2
            semi_true = cv2.imread(file_path, 0)
            semi_true = semi_true.astype(np.float64)
            # semi_true = semi_true*(79.16/255) - 127

            if dimension_length == None:
                dimension_length = semi_true.shape[0]

            if dimension_length == 'auto':
                dimension_length = math.floor(
                    semi_true.shape[0]/self.upscale) * self.upscale

            true = np.zeros((dimension_length, dimension_length))

            for y in range(dimension_length):
                for x in range(dimension_length):
                    true[y, x] = semi_true[y, x]

        else:
            raise ValueError("So is the file_type wireless-insite or image?")

        self.true_data = true
