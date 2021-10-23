# Author: fddf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator, griddata
from typing import Tuple, Callable, List
import time
import copy


def analyticalPreisachFunction1(a: float, b: float, c: float, d: float, n: float, p: float, q: float, beta: np.ndarray,
                                alpha: np.ndarray) -> np.ndarray:
    """
    Function based on Paper IEEE TRANSACTIONS ON MAGNETICS, VOL. 39, NO. 3, MAY 2003 'Analytical  Approximation  of  Preisach
    Distribution Functions' by Janos Fuezi
    """
    hm = (alpha + beta) / 2
    hc = (alpha - beta) / 2
    nom1 = c
    den1 = (1 + np.square(a) * np.square(alpha + b)) * (1 + np.square(a) * np.square(beta - b))
    nom2 = d
    den2 = np.exp(n * np.square(hm)) * np.exp(p * np.square(hc + q))
    preisach = nom1 / den1 + nom2 / den2
    # set lower right diagonal to zero
    for i in range(preisach.shape[0]):
        preisach[i, (-i - 1):] = 0
    return preisach


def analyticalPreisachFunction2(A: float, Hc: float, sigma: float, beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Function based on Paper 'Removing numerical instabilities in the Preisach model identification
    using genetic algorithms' by G. Consolo G. Finocchio, M. Carpentieri, B. Azzerboni.
    """
    nom1 = 1
    den1 = 1 + ((beta - Hc) * sigma / Hc) ** 2
    nom2 = 1
    den2 = 1 + ((alpha + Hc) * sigma / Hc) ** 2
    preisach = A * (nom1 / den1) * (nom2 / den2)
    # set lower right diagonal to zero
    for i in range(preisach.shape[0]):
        preisach[i, (-i - 1):] = 0
    return preisach


def initPreisachWithOnes(gridX: np.ndarray) -> np.ndarray:
    """
    Initialize the Preisach distribution function with ones over the entire Preisach-plane
    """
    preisach = np.ones_like(gridX)
    # set lower right diagonal to zero
    for i in range(preisach.shape[0]):
        preisach[i, (-i - 1):] = 0
    return preisach


def removeInBetween(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for removing in between points of an array
    """
    whipeout_indexes = np.empty(len(arr), dtype=np.bool)
    if len(arr) < 3:
        whipeout_indexes[:] = True
        return arr, whipeout_indexes
    else:
        range_list = list(range(1, len(arr) - 1))
        whipeout_indexes[0] = True
        whipeout_indexes[-1] = True
        for i in range_list:
            if arr[i] == arr[i - 1] == arr[i + 1]:
                whipeout_indexes[i] = False
            else:
                whipeout_indexes[i] = True
        return arr[whipeout_indexes], whipeout_indexes


def removeRedundantPoints(pointsX: np.ndarray, pointsY: np.ndarray) -> np.ndarray:
    """
    Function for removing redundant points inside vertices and horizontal lines of staircase polylines
    """
    pointsX, whipeout_indices = removeInBetween(pointsX)
    pointsY = pointsY[whipeout_indices]
    pointsY, whipeout_indices = removeInBetween(pointsY)
    pointsX = pointsX[whipeout_indices]
    return pointsX, pointsY


def preisachIntegration(w: float, Z: np.ndarray) -> np.ndarray:
    """
    Perform 2D- integration of the Preisach distribution function.
    """
    flipped = np.fliplr(np.flipud(w * Z))
    flipped_integral = np.cumsum(np.cumsum(flipped, axis=0), axis=1)
    return np.fliplr(np.flipud(flipped_integral))


class PreisachModel:
    """
    Efficient implementation of the scalar Preisach model
    """

    def __init__(self, n: int, alpha0: float):
        self.n = n
        self.alpha0 = alpha0
        self.beta0 = alpha0
        x = np.linspace(-self.beta0, self.beta0, n - 1)
        y = np.linspace(-self.alpha0, self.alpha0, n - 1)
        self.width = 2 * alpha0 / (n - 1)
        self.gridX, self.gridY = np.meshgrid(x, y)
        # flip  gridY to be compatible with definiton of preisach plane
        self.gridY = np.flipud(self.gridY)
        self.pointsX = np.array([-self.beta0], dtype=np.float64)
        self.pointsY = np.array([-self.alpha0], dtype=np.float64)
        self.interfaceX = np.array([-self.beta0, -self.beta0], dtype=np.float64)
        self.interfaceY = np.array([-self.alpha0, -self.alpha0], dtype=np.float64)
        self.historyInterfaceX: List[float] = []
        self.historyInterfaceY: List[float] = []
        self.historyU = - self.alpha0 * np.ones(1, dtype=np.float64)
        self.historyOut = np.zeros(0, dtype=np.float64)
        self.state = 'ascending'
        self.stateOld = 'ascending'
        self.stateChanged = False
        self.everett: Callable[[float, float], float]

    def __call__(self, *args, **kwargs):
        """
        Call model with input value given as argument
        """
        self.pointsX = self.interfaceX[:-1]
        self.pointsY = self.interfaceY[:-1]
        u = args[0]
        if u > self.historyU[-1]:
            self.state = 'ascending'
        elif u < self.historyU[-1]:
            self.state = 'descending'
        if self.state != self.stateOld:
            self.stateChanged = True
        else:
            self.stateChanged = False

        if self.stateChanged:
            # reached boundary
            self.pointsX = np.append(self.pointsX, self.historyU[-1])
            self.pointsY = np.append(self.pointsY, self.historyU[-1])

        if self.state == 'ascending':
            self.pointsY[self.pointsY <= u] = u
            self.pointsY[-1] = u

        elif self.state == 'descending':
            self.pointsX[self.pointsX >= u] = u
            self.pointsX[-1] = u

        self.interfaceX = np.append(self.pointsX, u)
        self.interfaceY = np.append(self.pointsY, u)

        self.interfaceX, self.interfaceY = removeRedundantPoints(self.interfaceX, self.interfaceY)

        self.stateOld = self.state
        self.historyInterfaceX.append(copy.deepcopy(self.interfaceX))
        self.historyInterfaceY.append(copy.deepcopy(self.interfaceY))
        self.historyU = np.append(self.historyU, copy.deepcopy(u))
        output = self.calculateOutput()
        self.historyOut = np.append(self.historyOut, copy.deepcopy(output))
        return output

    def setNegSatState(self):
        """
        Set the interface to negative saturation.
        """
        self.pointsX = np.array([-self.beta0], dtype=np.float64)
        self.pointsY = np.array([-self.alpha0], dtype=np.float64)
        self.interfaceX = np.array([-self.beta0, -self.beta0], dtype=np.float64)
        self.interfaceY = np.array([-self.alpha0, -self.alpha0], dtype=np.float64)
        self.resetHistory()

    def resetHistory(self):
        """
        Reset all model history parameters.
        """
        self.historyInterfaceX = []
        self.historyInterfaceY = []
        self.historyU = - self.alpha0 * np.ones(1, dtype=np.float64)
        self.historyOut = np.zeros(0, dtype=np.float64)
        self.state = 'ascending'
        self.stateOld = 'ascending'
        self.stateChanged = False

    def setDemagState(self, n: int = -1):
        """
        Function for setting the interface so that the output of the model will
        be zero initially (demagnetized state).

        Parameters
        ----------
        n : int
            Demagnetization step granularity
        """
        if n == -1:
            n = 150

        self.setNegSatState()
        excitation = np.linspace(1, 0, n)
        excitation[1::2] = - excitation[1::2]
        for i in excitation:
            self(i)

        self.resetHistory()

    def invert(self):
        """
        Return inverse Preisachmodel by constructing the inverse Everett
        function from the non inverted model using first order reversal curves (FODs).

        Grid points of the inverse Everett function are defined by the response values
        of the non inverse model. Inverse Everett function values on these grid points are directly
        defined by the dominant input extrema of the non inverted model. Inverse Everett function
        on irregular grid is interpolated using irregular grid interpolation.

        For a description of Preisach model inversion also see the following paper:
        'Identification and Inversion of Magnetic Hysteresis for Sinusoidal Magnetization' by Martin Kozek and
        Bernhard Gross
        """
        invModel = PreisachModel(self.n, self.alpha0)

        # Construct set of first order reversal curves (FODs) for identification of the inverse everett map
        # number of FODs correspond to the number of Hystereon elements n in Preisach plane
        FODs = np.zeros((self.n * self.n // 2 + self.n // 2 + 1, 2), dtype=np.float64)
        Mk = np.zeros(FODs.shape[0], dtype=np.float64)
        mk = np.zeros(FODs.shape[0], dtype=np.float64)
        invEverettVals = np.zeros(FODs.shape[0], dtype=np.float64)
        cnt = 0
        print('Inverting Model...')
        for valAlpha in np.linspace(-self.alpha0, self.alpha0, self.n - 1):
            for valBeta in np.linspace(-self.beta0, valAlpha, int((valAlpha - (-self.alpha0)) // self.width)):
                FODs[cnt, 0] = valAlpha
                FODs[cnt, 1] = valBeta
                # Reset and excite non inverted model with the FODs to get the grid Points of the inverse model
                # by dominant output extrema of the non inverted model
                self.setNegSatState()
                invEverettVals[cnt] = (1 / 2) * (valAlpha - valBeta)
                Mk[cnt] = self(valAlpha)
                mk[cnt] = self(valBeta)
                cnt += 1

        points = np.zeros((len(Mk), 2), dtype=np.float64)
        points[:, 1] = np.concatenate([Mk])
        points[:, 0] = np.concatenate([mk])
        Z = np.concatenate([invEverettVals])
        # Fit interpolator function on irregular grid using linear interpolation
        invEverettInterp = LinearNDInterpolator(points, Z, fill_value=0)

        # Set interpolator as everett function of the inverse model
        invModel.setEverettFunction(invEverettInterp)

        # return inverse model
        print('Model inversion succesfull !!!')
        return invModel

    def calculateOutput(self, **kwargs) -> float:
        """
        Calculate the output of the model with the current interface.
        Negative beta0 required, because beta0 was defined to be positive,
        however in the book 'mathematical models of hysteresis' from Mayergoyz, -
        (alpha0, beta0) is defined as the left top corner of the preisach triangle.
        Therefore beta0 has to be inverted to give the correct value
        Also the parameter order was defined different E(x,y)
        """
        if kwargs.get('mode'):
            mode = kwargs['mode'].lower()
        else:
            mode = 'default'

        if mode == 'default':
            sum = 0.0
            for i in range(1, len(self.interfaceX)):
                Mk = self.interfaceY[i]
                mk = self.interfaceX[i]
                mkOld = self.interfaceX[i - 1]
                sum = sum + (self.everett(mkOld, Mk) - self.everett(mk, Mk))
            output = -self.everett(-self.beta0, self.alpha0) + 2 * sum

        else:
            # alternative output calculation
            pass

        return output

    def setEverettFunction(self, everett: Callable[[float, float], float]):
        """
        Set everett function to given interpolator function.

        Parameters
        ----------
        everett : callable python method
            Interpolator for Everett function

        """
        if not isinstance(everett, collections.abc.Callable):
            raise ValueError('Given Parameter must be a callable function')
        self.everett = everett

    def showEverettFunction(self, fig: plt.Figure):
        """
        Show the Everett function in custom figure provided as argument
        """
        ax = fig.add_subplot(111, projection='3d')
        Z = self.everett(self.gridX, self.gridY)
        ax.plot_surface(gridX, gridY, Z)
        ax.set_title('Everett Function interpolated on regular grid')
        ax.set_xlabel('beta')
        ax.set_ylabel('alpha')
        ax.set_zlabel('z')
        plt.show()

    def showInterface(self, fig: plt.Figure):
        """
        Show the current  interface in custom figure provided as argument
        """
        ax = fig.add_subplot(111)
        ax.plot(self.interfaceX, self.interfaceY, 'r', linewidth=3)
        ax.plot(np.array([-self.beta0, self.beta0, -self.beta0, -self.beta0]),
                np.array([-self.alpha0, self.alpha0, self.alpha0, -self.alpha0]), linewidth=3)
        ax.xlim(-self.beta0 * 1.1, self.beta0 * 1.1)
        ax.ylim(-self.alpha0 * 1.1, self.alpha0 * 1.1)
        ax.title('{},{}'.format(self.interfaceX.tolist(), self.interfaceY.tolist()))
        ax.xlabel('Beta coefficients')
        ax.ylabel('Alpha coefficients')
        ax.axes().set_aspect('equal')
        ax.grid()
        ax.legend(['Interface', 'Preisach Plane'])
        ax.show()

    def animateHysteresis(self):
        # @Todo vector length of u and out must be same
        self.historyU = self.historyU[1:]

        def update_line(num, self, line1, line2, line3):
            line1.set_xdata(num)
            line1.set_ydata(self.historyU[num])
            line2.set_xdata(self.historyInterfaceX[num])
            line2.set_ydata(self.historyInterfaceY[num])
            line3.set_xdata(self.historyU[num])
            line3.set_ydata(self.historyOut[num])
            return line1, line2, line3

        frames = len(self.historyInterfaceX)

        gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 1, 1])
        fig1 = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])

        # create plot of input
        ax1.plot(self.historyU, linewidth=1)
        ax1.set_xlim([0, len(self.historyU)])
        ax1.set_ylim([-self.alpha0 * 1.1, self.alpha0 * 1.1])
        ax1.set_xlabel('samples')
        ax1.set_ylabel('input')

        line1, = ax1.plot([0.0], [0.0], '.', markersize=15)

        # create plot of preisach plane
        ax2.plot(np.array([-self.beta0, self.beta0, -self.beta0, -self.beta0]),
                 np.array([-self.alpha0, self.alpha0, self.alpha0, -self.alpha0]), linewidth=3)
        line2, = ax2.plot([], [], 'r', linewidth=2)
        ax2.set_xlim(-self.beta0 * 1.1, self.beta0 * 1.1)
        ax2.set_ylim(-self.alpha0 * 1.1, self.alpha0 * 1.1)
        ax2.set_xlabel('Beta coefficients')
        ax2.set_ylabel('Alpha coefficients')
        ax2.legend(['Preisach plane', 'Interface'], loc='lower right')

        # create plot of hysteresis
        ax3.plot(self.historyU, self.historyOut)
        line3, = ax3.plot([0.0], [0.0], '.', markersize=15)

        simulation = animation.FuncAnimation(fig1, update_line, frames,
                                             fargs=(self, line1, line2, line3), interval=25,
                                             blit=True, repeat=False)

        plt.show()
        return simulation


if __name__ == "__main__":

    model = PreisachModel(200, 1)
    gridX = model.gridX
    gridY = model.gridY
    width = model.width

    ######## init with ones #########
    # preisach = initPreisachWithOnes()

    # ####### analytic 1 #############
    # A = 71
    # B = -0.018
    # C = 0.013
    # D = 0.068
    # N = 15
    # P = 2500
    # Q = 0.04
    # preisach = analyticalPreisachFunction1(A, B, C, D, N, P, Q, gridX, gridY)

    ######## analytic 2 #############
    A = 1
    Hc = 0.01
    sigma = 0.03
    preisach = analyticalPreisachFunction2(A, Hc, sigma, gridX, gridY)

    # Calculate Everett function from preisach function
    everett = preisachIntegration(width, preisach)

    # Scale Everett function to a maximum value of 1
    everett = everett / np.max(everett)

    # Calculate linear Interpolator for Everett function
    points = np.zeros((everett.size, 2), dtype=np.float64)
    points[:, 0] = gridX.flatten()
    points[:, 1] = gridY.flatten()
    values = everett.flatten()
    everettInterp = LinearNDInterpolator(points, values)
    model.setEverettFunction(everettInterp)

    # show everett function of model
    fig = plt.figure()
    model.showEverettFunction(fig)

    # calculate inverse model
    invModel = model.invert()

    # show everett function of inverse model
    fig = plt.figure()
    invModel.showEverettFunction(fig)

    # Create excitation signal
    nSamps = 2500
    phi = np.linspace(0, 2 * np.pi + np.pi / 2, nSamps)

    sawtooth = np.zeros(nSamps, dtype=np.float64)
    sawtooth[phi < np.pi / 2] = 0.7 * 2 / np.pi * phi[phi < np.pi / 2]
    sawtooth[np.logical_and(phi < 3 * np.pi / 2, phi > np.pi / 2)] = -0.7 * 2 / np.pi * (
            phi[np.logical_and(phi < 3 * np.pi / 2, phi > np.pi / 2)] - np.pi)
    sawtooth[phi > 3 * np.pi / 2] = 0.7 * 2 / np.pi * (phi[phi > 3 * np.pi / 2] - 2 * np.pi)

    input = 0.15 * np.sin(30 * phi) + sawtooth
    output = np.zeros_like(input, dtype=np.float64)
    middle = np.zeros_like(input, dtype=np.float64)

    model.setDemagState(80)
    invModel.setDemagState(80)

    # Apply input to inverse model and then apply it to non inverse model
    for i in range(len(input)):
        middle[i] = model(input[i])
        output[i] = invModel(middle[i])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(input)
    # ax.plot(middle)
    # ax.plot(output)
    # ax.legend(['input', 'middle', 'output'])
    # plt.show()

    simulation1 = model.animateHysteresis()
    simulation2 = invModel.animateHysteresis()
    # Uncomment the next line if you want to save the animation
    # simulation1.save(filename='hysterese_simulation.mp4', fps=30, dpi=300)
    # simulation2.save(filename='hysterese_invertiert_simulation.mp4', fps=30, dpi=300)
