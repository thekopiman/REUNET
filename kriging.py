import numpy as np
import openturns as ot
from time import perf_counter
import math
from base import BaseRM
import cv2


class Kriging(BaseRM):
    def __init__(self):
        super().__init__()

    def secondstep(self, upscale=None, model='squaredexponential',) -> None:
        '''
        This module is meant for Simulation Test purposes only.
        '''
        self.upscale = upscale
        self.durations = {}
        variogram_start = perf_counter()
        coordinates = np.transpose(np.nonzero(self.sparsedata))
        input_train = ot.Sample(coordinates)
        observations = np.zeros((len(coordinates), 1))

        for i in range(len(coordinates)):
            observations[i, 0] = self.sparsedata[coordinates[i]
                                                 [0], coordinates[i][1]]

        output_train = ot.Sample(observations)
        inputDimension = 2
        basis = ot.ConstantBasisFactory(inputDimension).build()
        if model == 'absoluteexponential':
            covariance_kernel = ot.AbsoluteExponential(
                [1.0]*inputDimension, [1.0])
        elif model == 'diraccovariance':
            covariance_kernel = ot.DiracCovarianceModel(
                inputDimension)
        elif model == 'exponential':
            covariance_kernel = ot.ExponentialModel(
                [1.0]*inputDimension, [1.0])
        elif model == 'matern':
            covariance_kernel = ot.MaternModel(
                inputDimension)
        elif model == 'spherical':
            covariance_kernel = ot.SphericalModel([1.0]*inputDimension, [1.0])
        elif model == 'squaredexponential':
            covariance_kernel = ot.SquaredExponential(
                [1.0]*inputDimension, [1.0])
        else:
            raise AttributeError('Wrong model name')

        algo = ot.KrigingAlgorithm(input_train, output_train,
                                   covariance_kernel, basis)
        algo.run()
        result = algo.getResult()
        self.krigingMetamodel = result.getMetaModel()
        self.myInterval = ot.Interval(
            [0.0, 0.0], [self.sparsedata.shape[0], self.sparsedata.shape[1]])

        self.durations['variogram'] = perf_counter() - variogram_start

    def uniformity(self):
        self.matrix = self.sparsedata
        nx = self.matrix.shape[1]
        ny = self.matrix.shape[0]

        weights_calculation_start = perf_counter()
        myIndices = [ny - 1, nx - 1]
        myMesher = ot.IntervalMesher(myIndices)
        myMeshBox = myMesher.build(self.myInterval)
        vertices = myMeshBox.getVertices()

        predictions = self.krigingMetamodel(vertices)

        predictions_array = np.transpose(np.array(
            predictions).reshape(ny, nx))

        self.durations['uniformity'] = perf_counter() - \
            weights_calculation_start

        return predictions_array

    def purekriging(self):
        self.matrix = self.sparsedata
        nx = self.matrix.shape[1]
        ny = self.matrix.shape[0]

        weights_calculation_start = perf_counter()
        myIndices = [ny - 1, nx - 1]
        myMesher = ot.IntervalMesher(myIndices)
        myMeshBox = myMesher.build(self.myInterval)
        vertices = myMeshBox.getVertices()

        predictions = self.krigingMetamodel(vertices)

        predictions_array = np.transpose(np.array(
            predictions).reshape(ny, nx))

        self.durations['purekriging'] = perf_counter() - \
            weights_calculation_start

        return predictions_array

    def return_duations(self) -> dict:
        return self.durations

    def __getitem__(self):
        return self.purekriging()
