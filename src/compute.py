import torch
import ctypes
import decimal as d
from collections import OrderedDict
from scipy.constants import c
from scipy.constants import epsilon_0 as e0
import gc
from torch import Tensor
from src.multiscale import apply_filter
from src.tv import total_variation
from src import lib, libsr,libuc,libph,libpe,libback,z0
import time
import numpy as np
import torch.nn.functional as F

class GPRFWIModel(torch.nn.Module):

    def __init__(self, se:Tensor, er:Tensor, serequires_grad:bool=False, errequires_grad:bool=False):
        super().__init__()
        self.er = torch.nn.Parameter(er, requires_grad=errequires_grad)
        self.se = torch.nn.Parameter(se, requires_grad=serequires_grad)

    def forward(self, device, dx, ny, nx, nz, time_windows, source, source_location, source_step,
            receiver_location, receiver_step, pmlthick, source_apmlitudes, step, mr, nsrc, nrx, freq=None,
            dt=None,pml_formulation='HORIPML', total_step=None,tv=None):
        
        return compute(device, dx, ny, nx, nz, time_windows, source, source_location, source_step,
            receiver_location, receiver_step, pmlthick, source_apmlitudes, step, self.er, self.se, mr, nsrc, nrx, freq=freq,
            dt=dt,pml_formulation=pml_formulation, total_step=total_step,tv=tv)



def compute(device, dx, ny, nx, nz,source,source_apmlitudes, step, er, se, mr, nsrc, nrx,  
            time_windows=None, source_location=None, source_step=None,
            receiver_location=None, receiver_step=None, pmlthick=0,  freq=None,
            dt=None,pml_formulation='HORIPML', total_step=None,tv=True,
            Fixedreceiver=False,nseg=None,customsource=None,customreceiver=None):
    se = torch.clamp(se, min=0) 
    er = torch.clamp(er, min=1) 
    if nseg==None:
        G = initialization(device, dx, ny, nx, nz, time_windows, source, source_location, source_step,
                receiver_location, receiver_step, pmlthick, source_apmlitudes, step, er, se, mr, nsrc, nrx,
                dt, pml_formulation, freq, total_step,tv,Fixedreceiver,nseg,customsource,customreceiver)
        rxs_gpu = GPRFWI.apply(G, er, se, mr)
    else:
        G = initialization(device, dx, ny, nx, nz, time_windows, source, source_location, source_step,
                receiver_location, receiver_step, pmlthick, source_apmlitudes, step, er, se, mr, nsrc, nrx,
                dt, pml_formulation, freq, total_step,tv,Fixedreceiver,nseg)
        rxs_gpu = GPRFWI.apply(G, er, se, mr)
    return rxs_gpu, G.dt



class CFSParameter(object):
    scalingprofiles = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5, 'sextic': 6,
                       'septic': 7, 'octic': 8}
    scalingdirections = ['forward', 'reverse']

    def __init__(self, ID=None, scaling='polynomial', scalingprofile=None, scalingdirection='forward', min=0, max=0):
        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.scalingdirection = scalingdirection
        self.min = min
        self.max = max


class CFS(object):
    """CFS term for PML."""

    def __init__(self, scalingdirection='forward'):

        self.alpha = CFSParameter(ID='alpha', scalingprofile='constant', scalingdirection=scalingdirection)
        self.kappa = CFSParameter(ID='kappa', scalingprofile='constant', min=1, max=1,
                                  scalingdirection=scalingdirection)
        self.sigma = CFSParameter(ID='sigma', scalingprofile='quartic', min=0, max=None,
                                  scalingdirection=scalingdirection)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_sigmamax(self, d, er, mr, G):

        # Calculation of the maximum value of sigma from http://dx.doi.org/10.1109/8.546249
        m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
        self.sigma.max = (0.8 * (m + 1)) / (z0 * d * torch.sqrt(er * mr))

    def scaling_polynomial(self, order, Evalues, Hvalues):
        tmp = (torch.linspace(0, (len(Evalues) - 1) + 0.5, steps=2 * len(Evalues)) / (len(Evalues) - 1)) ** order
        Evalues = tmp[0:-1:2].to(self.device)
        Hvalues = tmp[1::2].to(self.device)
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):

        # Extra cell of thickness added to allow correct scaling of electric and magnetic values
        Evalues = torch.zeros(thickness + 1, device=self.device)
        Hvalues = torch.zeros(thickness + 1, device=self.device)
        if parameter.scalingprofile == 'constant':
            Evalues += parameter.max
            Hvalues += parameter.max
        elif parameter.scaling == 'polynomial':
            Evalues, Hvalues = self.scaling_polynomial(CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues,
                                                       Hvalues)
            if parameter.ID == 'alpha':
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == 'kappa':
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == 'sigma':
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        if parameter.scalingdirection == 'reverse':
            Evalues = Evalues.flip(dims=(0,))
            Hvalues = Hvalues.flip(dims=(0,))
            # Magnetic values must be shifted one element to the left after reversal
            Hvalues = torch.roll(Hvalues, -1)

        # Extra cell of thickness not required and therefore removed after scaling
        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]

        return Evalues, Hvalues


class PML(object):
    formulations = ['HORIPML', 'MRIPML']
    boundaryIDs = ['x0', 'y0', 'z0', 'xmax', 'ymax', 'zmax']
    directions = ['xminus', 'yminus', 'zminus', 'xplus', 'yplus', 'zplus']

    def __init__(self, G, ID=None, direction=None, xs=0, xf=0, ys=0, yf=0, zs=0, zf=0, adjoint=False):
        """
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
            ID (str): Identifier for PML slab.
            direction (str): Direction of increasing absorption.
            xs, xf, ys, yf, zs, zf (float): Extent of the PML slab.
        """
        self.ID = ID
        self.direction = direction
        self.xs = xs
        self.xf = xf
        self.ys = ys
        self.yf = yf
        self.zs = zs
        self.zf = zf
        self.nx = xf - xs
        self.ny = yf - ys
        self.nz = zf - zs
        self.device = G.device
        self.step = G.step
        self.adjoint = False
        # Spatial discretisation and thickness
        if self.direction[0] == 'x':
            self.d = G.dx
            self.thickness = self.nx
        elif self.direction[0] == 'y':
            self.d = G.dy
            self.thickness = self.ny
        elif self.direction[0] == 'z':
            self.d = G.dz
            self.thickness = self.nz

        self.CFS = G.cfs

    def calculate_update_coeffs(self, er, mr, G):

        self.ERA_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.ERB_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.ERE_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.ERF_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.HRA_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.HRB_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.HRE_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)
        self.HRF_gpu = torch.zeros((len(self.CFS), self.thickness), device=self.device)

        for x, cfs in enumerate(self.CFS):
            if not cfs.sigma.max:
                cfs.calculate_sigmamax(self.d, er, mr, G)
            if self.adjoint:
                cfs.alpha.scalingdirection = 'reverse'
                cfs.kappa.scalingdirection = 'reverse'
                cfs.sigma.scalingdirection = 'reverse'
                Ealpha, Halpha = cfs.calculate_values(self.thickness, cfs.alpha)
                Ekappa, Hkappa = cfs.calculate_values(self.thickness, cfs.kappa)
                Esigma, Hsigma = cfs.calculate_values(self.thickness, cfs.sigma)
            else:
                Ealpha, Halpha = cfs.calculate_values(self.thickness, cfs.alpha)
                Ekappa, Hkappa = cfs.calculate_values(self.thickness, cfs.kappa)
                Esigma, Hsigma = cfs.calculate_values(self.thickness, cfs.sigma)

            # Define different parameters depending on PML formulation
            if G.pmlformulation == 'HORIPML':
                # HORIPML electric update coefficients
                # Esigma=Esigma.to(G.device)
                # Hsigma = Hsigma.to(G.device)
                # dt=G.dt.to(G.device)
                if self.adjoint:
                    tmp = (2 * e0 * Ekappa) - G.dt * (Ealpha * Ekappa + Esigma)
                    self.ERA_gpu[x, :] = (2 * e0 - G.dt * Ealpha) / tmp
                    self.ERB_gpu[x, :] = (2 * e0 * Ekappa) / tmp
                    self.ERE_gpu[x, :] = ((2 * e0 * Ekappa) + G.dt * (Ealpha * Ekappa + Esigma)) / tmp
                    self.ERF_gpu[x, :] = (-2 * Esigma * G.dt) / (Ekappa * tmp)

                    tmp_adj = (2 * e0 * Hkappa) - G.dt * (Halpha * Hkappa + Hsigma)
                    self.HRA_gpu[x, :] = (2 * e0 - G.dt * Halpha) / tmp_adj
                    self.HRB_gpu[x, :] = (2 * e0 * Hkappa) / tmp_adj
                    self.HRE_gpu[x, :] = ((2 * e0 * Hkappa) + G.dt * (Halpha * Hkappa + Hsigma)) / tmp_adj
                    self.HRF_gpu[x, :] = (-2 * Hsigma * G.dt) / (Hkappa * tmp_adj)
                    # 同样修改磁场系数
                else:
                    tmp = (2 * e0 * Ekappa) + G.dt * (Ealpha * Ekappa + Esigma)
                    self.ERA_gpu[x, :] = (2 * e0 + G.dt * Ealpha) / tmp
                    self.ERB_gpu[x, :] = (2 * e0 * Ekappa) / tmp
                    self.ERE_gpu[x, :] = ((2 * e0 * Ekappa) - G.dt * (Ealpha * Ekappa + Esigma)) / tmp
                    self.ERF_gpu[x, :] = (2 * Esigma * G.dt) / (Ekappa * tmp)
                    # HORIPML magnetic update coefficients
                    tmp = (2 * e0 * Hkappa) + G.dt * (Halpha * Hkappa + Hsigma)
                    self.HRA_gpu[x, :] = (2 * e0 + G.dt * Halpha) / tmp
                    self.HRB_gpu[x, :] = (2 * e0 * Hkappa) / tmp
                    self.HRE_gpu[x, :] = ((2 * e0 * Hkappa) - G.dt * (Halpha * Hkappa + Hsigma)) / tmp
                    self.HRF_gpu[x, :] = (2 * Hsigma * G.dt) / (Hkappa * tmp)

    def gpu_initialise_arrays(self):
        """Initialise PML field and coefficient arrays on GPU."""
        if self.direction[0] == 'x':
            self.EPhi1_gpu = torch.zeros((G.step, self.nx + 1, self.ny, self.nz + 1),
                                         device=self.device)
            self.EPhi1_gpu = self.EPhi1_gpu.contiguous()
            self.EPhi2_gpu = torch.zeros((G.step, self.nx + 1, self.ny + 1, self.nz),
                                         device=self.device)
            self.EPhi2_gpu = self.EPhi2_gpu.contiguous()
            self.HPhi1_gpu = torch.zeros((G.step, self.nx, self.ny + 1, self.nz),
                                         device=self.device)
            self.HPhi1_gpu = self.HPhi1_gpu.contiguous()
            self.HPhi2_gpu = torch.zeros((G.step, self.nx, self.ny, self.nz + 1),
                                         device=self.device)
            self.HPhi2_gpu = self.HPhi2_gpu.contiguous()
        elif self.direction[0] == 'y':
            self.EPhi1_gpu = torch.zeros((G.step, self.nx, self.ny + 1, self.nz + 1),
                                         device=self.device)
            self.EPhi1_gpu = self.EPhi1_gpu.contiguous()
            self.EPhi2_gpu = torch.zeros((G.step, self.nx + 1, self.ny + 1, self.nz),
                                         device=self.device)
            self.EPhi2_gpu = self.EPhi2_gpu.contiguous()
            self.HPhi1_gpu = torch.zeros((G.step, self.nx + 1, self.ny, self.nz),
                                         device=self.device)
            self.HPhi1_gpu = self.HPhi1_gpu.contiguous()
            self.HPhi2_gpu = torch.zeros((G.step, self.nx, self.ny, self.nz + 1),
                                         device=self.device)
            self.HPhi2_gpu = self.HPhi2_gpu.contiguous()
        elif self.direction[0] == 'z':
            self.EPhi1_gpu = torch.zeros((G.step, self.nx, self.ny + 1, self.nz + 1),
                                         device=self.device)
            self.EPhi1_gpu = self.EPhi1_gpu.contiguous()
            self.EPhi2_gpu = torch.zeros((G.step, self.nx + 1, self.ny, self.nz + 1),
                                         device=self.device)
            self.EPhi2_gpu = self.EPhi2_gpu.contiguous()
            self.HPhi1_gpu = torch.zeros((G.step, self.nx + 1, self.ny, self.nz),
                                         device=self.device)
            self.HPhi1_gpu = self.HPhi1_gpu.contiguous()
            self.HPhi2_gpu = torch.zeros((G.step, self.nx, self.ny + 1, self.nz),
                                         device=self.device)
            self.HPhi2_gpu = self.HPhi2_gpu.contiguous()


def build_pmls(G, er, se, mr):
    for key, value in G.pmlthickness.items():
        if value > 0:
            sumer = 0  # Sum of relative permittivities in PML slab
            summr = 0  # Sum of relative permeabilities in PML slab
            if key[0] == 'x':
                if key == 'x0':
                    pml = PML(G, ID=key, direction='xminus', xf=value, yf=G.ny, zf=G.nz)
                elif key == 'xmax':
                    pml = PML(G, ID=key, direction='xplus', xs=G.nx - value, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for j in range(G.ny):
                    for k in range(G.nz):
                        sumer += er[pml.xs, j, k]
                        summr += mr[pml.xs, j, k]
                averageer = sumer / (G.ny * G.nz)
                averagemr = summr / (G.ny * G.nz)

            elif key[0] == 'y':
                if key == 'y0':
                    pml = PML(G, ID=key, direction='yminus', yf=value, xf=G.nx, zf=G.nz)
                elif key == 'ymax':
                    pml = PML(G, ID=key, direction='yplus', ys=G.ny - value, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for i in range(G.nx):
                    for k in range(G.nz):
                        sumer += er[i, pml.ys, k]
                        summr += mr[i, pml.ys, k]

                averageer = sumer / (G.nx * G.nz)
                averagemr = summr / (G.nx * G.nz)

            elif key[0] == 'z':
                if key == 'z0':
                    pml = PML(G, ID=key, direction='zminus', zf=value, xf=G.nx, yf=G.ny)
                elif key == 'zmax':
                    pml = PML(G, ID=key, direction='zplus', zs=G.nz - value, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for i in range(G.nx):
                    for j in range(G.ny):
                        sumer += er[i, j, pml.zs]
                        summr += mr[i, j, pml.zs]

                averageer = sumer / (G.nx * G.ny)
                averagemr = summr / (G.nx * G.ny)

            pml.calculate_update_coeffs(averageer, averagemr, G)


# In[5]:


class Source(object):
    """Super-class which describes a generic source."""

    def __init__(self):
        self.polarisation = None
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordorigin = None
        self.ycoordorigin = None
        self.zcoordorigin = None
        self.start = None
        self.stop = None
        self.waveformID = None

    def calculate_waveform_values(self, G):

        # Waveform values on timesteps
        self.waveformvalues_wholestep = torch.zeros((G.iterations), device=G.device)
        # Waveform values on half timesteps
        self.waveformvalues_halfstep = torch.zeros((G.iterations), device=G.device)
        waveform = next(x for x in G.waveforms)
        if waveform.type == 'user':
            # ampvalue1=np.load('lnapl/lnaplsource.npy')
            # ampvalue=torch.tensor(ampvalue1).unsqueeze(0).unsqueeze(0)
            # x_upsampled = F.interpolate(ampvalue, scale_factor=4, mode='linear', align_corners=True)
            # x_upsampled = x_upsampled.squeeze(0).squeeze(0)
            # self.waveformvalues_wholestep[:x_upsampled.shape[0]] = x_upsampled
            ampvalue1=np.load('../lnapl/lnaplsource.npy')
            ampvalue=torch.tensor(ampvalue1)
            self.waveformvalues_wholestep[:ampvalue.shape[0]] = ampvalue
        else:
            for iteration in range(G.iterations):
                time = G.dt * (iteration)
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any delay in the start
                    time -= self.start
                    self.waveformvalues_wholestep[iteration] = waveform.calculate_value(time, G.dt)
                    self.waveformvalues_halfstep[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)


class HertzianDipole(Source):
    """A Hertzian dipole is an additive source (electric current density)."""

    def __init__(self):
        super().__init__()
        self.dl = None




class Waveform(object):
    types = ['gaussian', 'gaussiandot', 'gaussiandotnorm', 'gaussiandotdot', 'gaussiandotdotnorm', 'gaussianprime',
             'gaussianfloatprime', 'ricker', 'sine', 'contsine', 'impulse', 'user']

    def __init__(self):
        self.type = None
        self.amp = 1
        self.freq = None
        self.userfunc = None
        self.chi = 0
        self.zeta = 0
        self.delay = 0

    def calculate_coefficients(self):
        if self.type in ['gaussian', 'gaussiandot', 'gaussiandotnorm', 'gaussianprime', 'gaussianfloatprime']:
            self.chi = 1 / self.freq
            self.zeta = 2 * (torch.pi ** 2) * (self.freq ** 2)
        elif self.type in ['gaussiandotdot', 'gaussiandotdotnorm', 'ricker']:
            self.chi = torch.sqrt(torch.tensor(2.0)) / self.freq
            self.zeta = (torch.pi ** 2) * (self.freq ** 2)

    def calculate_value(self, time, dt):
        self.calculate_coefficients()
        if self.type == 'ricker':
            delay = time - self.chi
            normalise = 1 / (2 * self.zeta)
            ampvalue = -(2 * self.zeta * (2 * self.zeta * delay ** 2 - 1) * torch.exp(
                -self.zeta * delay ** 2)) * normalise

        ampvalue *= self.amp
        return ampvalue


class Grid(object):
    """Generic grid/mesh."""

    def __init__(self, grid, device):
        self.nx = grid.shape[0]
        self.ny = grid.shape[1]
        self.nz = grid.shape[2]
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.grid = grid
        self.device = device

    def n_edges(self):
        i = self.nx
        j = self.ny
        k = self.nz
        e = (i * j * (k - 1)) + (j * k * (i - 1)) + (i * k * (j - 1))
        return e

    def n_nodes(self):
        return self.nx * self.ny * self.nz

    def n_cells(self):
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1)

    def get(self, i, j, k):
        return self.grid[i, j, k]

    def within_bounds(self, **kwargs):
        for co, val in kwargs.items():
            if val < 0 or val > getattr(self, 'n' + co):
                raise ValueError(co)

    def calculate_coord(self, coord, val):
        co = round_value(torch.tensor(val, dtype=torch.float) / getattr(self, 'd' + coord))
        return co


class FDTDGrid(Grid):
    def __init__(self):
        self.outputdirectory = ''
        self.title = ''
        # Threads per block - electric and magnetic field updates
        self.tpb = (256, 1, 1)
        self.device = ''
        self.highestfreqthres = 40
        self.maxnumericaldisp = 2
        self.mingridsampling = 3
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.mode = None
        self.iterations = 0
        self.timewindow = 0
        self.pmlthickness = OrderedDict((key, 10) for key in PML.boundaryIDs)
        self.cfs = []
        self.pmls = []
        self.pmlformulation = 'HORIPML'
        self.materials = []
        self.averagevolumeobjects = True
        self.fractalvolumes = []
        self.waveforms = []
        self.hertziandipoles = []
        self.magneticdipoles = []
        self.transmissionlines = []
        self.rxs = None
        self.source = None
        self.receiver = None
        self.step = 1
        self.pmlthick = 0
        self.freq = None
        self.total_step = None
        self.tv=None
        self.nseg=None

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE0 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)

        self.updatecoeffsE1 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)
        self.updatecoeffsE4 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)
        self.updatecoeffsH0 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)
        self.updatecoeffsH1 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)
        self.updatecoeffsH4 = torch.zeros((self.nx + 1, self.ny + 1, self.nz + 1),
                                          device=self.device)

    def gpu_initialise_arrays(self):
        self.Ex_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Ex_gpu = self.Ex_gpu.contiguous()
        self.Ey_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Ey_gpu = self.Ey_gpu.contiguous()
        self.Hz_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Hz_gpu = self.Hz_gpu.contiguous()

        self.Ez_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Ez_gpu = self.Ez_gpu.contiguous()
        self.Hx_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Hx_gpu = self.Hx_gpu.contiguous()
        self.Hy_gpu = torch.zeros((self.step, self.nx + 1, self.ny + 1, self.nz + 1),
                                  device=self.device)
        self.Hy_gpu = self.Hy_gpu.contiguous()


def get_other_directions(direction):
    directions = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
    return directions[direction]


def round_value(value, decimalplaces=0):
    # 如果输入值是 Tensor 类型，转换为 float 类型
    if isinstance(value, torch.Tensor):
        value = value.item()
    # Rounds to nearest integer (half values are rounded downwards)
    if decimalplaces == 0:
        rounded = int(d.Decimal(value).quantize(d.Decimal('1'), rounding=d.ROUND_HALF_DOWN))
    # Rounds down to nearest float represented by number of decimal places
    else:
        precision = '1.{places}'.format(places='0' * decimalplaces)
        rounded = torch.tensor(d.Decimal(value).quantize(d.Decimal(precision), rounding=d.ROUND_FLOOR),
                               dtype=torch.float)
    return rounded
import copy


def Forward(G,iseg=None):
    if G.nseg!=None:
        Gsave=[]
        seg = (G.iterations + G.nseg - 1) // G.nseg+1
        Gsavenumber=[]
        for i in range(G.nseg-1):
            Gsavenumber.append((G.iterations-(i+1)*seg-1))
        Gsavenumber.append(0)
        segsave=G.iterations-seg
        Ez=torch.zeros((seg,G.step, G.nx , G.ny , G.nz), device=G.device)
    else:
        Ez=torch.zeros((G.iterations,G.step, G.nx , G.ny , G.nz), device=G.device)
    sourcei = G.source
    receiveri = G.receiver
    rxs_gpu = torch.zeros((G.step, 6, G.iterations, G.nrx),
                          device=G.device)

    rxs_ptr = rxs_gpu.data_ptr()

    srcwaves_gpu = torch.zeros((G.nsrc, G.iterations), device=G.device)
    srcinfo1_ptr = sourcei.data_ptr()
    for i, src in enumerate(G.hertziandipoles):
        srcwaves_gpu[i, :] = src.waveformvalues_wholestep
        if src.polarisation == 'z':
            polar = 2
        elif src.polarisation == 'x':
            polar = 0
        elif src.polarisation == 'y':
            polar = 1
    srcinfo2_gpu = G.dz

    gs=-1
    for iteration in range(G.iterations):
        if G.nseg!=None:
            if iteration==Gsavenumber[-gs]:
                Gsave.append(copy.deepcopy(G))
        libsr.launch_store_outputs(
            G.step,
            G.nrx,
            iteration,
            ctypes.cast(receiveri.data_ptr(), ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(rxs_ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            G.nx + 1,
            G.ny + 1,
            G.nz + 1,
            3,
            6,
            G.iterations
        )

        # 2. Update magnetic field components
        lib.h_fields_updates(
            ctypes.cast(G.updatecoeffsH0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.updatecoeffsH1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            G.step,
            G.nx + 1,  # a
            G.ny + 1,  # b
            G.nz + 1)

        # 3.Update magnetic field components with the PML correction
        for pml in G.pmls:
            funcphh = getattr(libph, pml.update_magnetic_gpu)
            funcphh.argtypes = libph.argtypes
            funcphh.restype = libph.restype

            HPhi1_gpu_ptr = pml.HPhi1_gpu.data_ptr()
            HPhi2_gpu_ptr = pml.HPhi2_gpu.data_ptr()
            HRA_gpu_ptr = pml.HRA_gpu.data_ptr()
            HRB_gpu_ptr = pml.HRB_gpu.data_ptr()
            HRE_gpu_ptr = pml.HRE_gpu.data_ptr()
            HRF_gpu_ptr = pml.HRF_gpu.data_ptr()

            funcphh(pml.xs, pml.xf, pml.ys,
                    pml.yf, pml.zs, pml.zf,
                    pml.HPhi1_gpu.shape[1], pml.HPhi1_gpu.shape[2], pml.HPhi1_gpu.shape[3],
                    pml.HPhi2_gpu.shape[1], pml.HPhi2_gpu.shape[2], pml.HPhi2_gpu.shape[3],
                    pml.thickness,
                    ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HPhi1_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HPhi2_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRA_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRB_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRE_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRF_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    pml.d, ctypes.cast(G.updatecoeffsH4.data_ptr(), ctypes.POINTER(ctypes.c_float)), G.nx + 1, G.ny + 1, G.nz + 1, G.step)

        lib.e_fields_updates(
            ctypes.cast(G.updatecoeffsE0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.updatecoeffsE1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            G.step,
            G.nx + 1,  # a
            G.ny + 1,  # b
            G.nz + 1)  # c

        for pml in G.pmls:
            funcphe = getattr(libpe, pml.update_electric_gpu)
            funcphe.argtypes = libpe.argtypes
            funcphe.restype = libpe.restype

            EPhi1_gpu_ptr = pml.EPhi1_gpu.data_ptr()
            EPhi2_gpu_ptr = pml.EPhi2_gpu.data_ptr()
            ERA_gpu_ptr = pml.ERA_gpu.data_ptr()
            ERB_gpu_ptr = pml.ERB_gpu.data_ptr()
            ERE_gpu_ptr = pml.ERE_gpu.data_ptr()
            ERF_gpu_ptr = pml.ERF_gpu.data_ptr()

            funcphe(pml.xs, pml.xf, pml.ys,
                    pml.yf, pml.zs, pml.zf,
                    pml.EPhi1_gpu.shape[1], pml.EPhi1_gpu.shape[2], pml.EPhi1_gpu.shape[3],
                    pml.EPhi2_gpu.shape[1], pml.EPhi2_gpu.shape[2], pml.EPhi2_gpu.shape[3],
                    pml.thickness,
                    ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(G.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(EPhi1_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(EPhi2_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERA_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERB_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERE_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERF_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    pml.d, ctypes.cast(G.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)), G.nx + 1, G.ny + 1, G.nz + 1, G.step)

        libsr.update_hertzian_dipole(
            G.step,
            iteration,
            G.dx, G.dy, G.dz,
            ctypes.cast(srcinfo1_ptr, ctypes.POINTER(ctypes.c_int)),
            srcinfo2_gpu,
            ctypes.cast(srcwaves_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(G.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            G.nx + 1,
            G.ny + 1,
            G.nz + 1,
            G.nsrc,
            polar,
            G.iterations
        )
        if G.nseg!=None:
            if iteration>=segsave:
                Ez[iteration-segsave, :, :, :, :] = G.Ez_gpu[:, :G.nx, :G.ny, :G.nz]
        else:
            Ez[iteration, :, :, :, :] = G.Ez_gpu[:, :G.nx, :G.ny, :G.nz]
    # visualize_field(G, iteration,save_dir='field_visualizations/forward',reverse=False)

    gc.collect()
    torch.cuda.empty_cache()
    if  G.nseg!=None:
        return rxs_gpu, Ez,Gsave
    else:
        return rxs_gpu, Ez,None


# In[9]:


def initialization(device, dx, ny, nx, nz, time_windows, source, source_location, source_step,
        receiver_location, receiver_step, pmlthick, source_apmlitudes, step, er, se, mr, nsrc, nrx, dt, pml_formulation,
        freq, total_step,tv,Fixedreceiver,nseg,customsource,customreceiver):
    se = torch.clamp(se, min=0, max=1e6) 
    global G
    G = FDTDGrid()
    G.tv=tv
    if total_step != None:
        G.total_step = total_step
    if nseg != None:
        G.nseg = nseg
    G.freq = freq
    G.step = step
    G.pmlformulation = pml_formulation
    G.device = device
    G.dx = dx
    G.dy = dx
    G.dz = dx
    # G.nx = round_value(nx / G.dx)
    # G.ny = round_value(ny / G.dy)
    # G.nz = round_value(nz / G.dz)
    G.nx = nx
    G.ny = ny
    G.nz = nz
    G.nrx = nrx
    G.nsrc = nsrc
    if dt == None:
        if G.nz:
            G.dt = 1 / (c * ((((1 / G.dx) * (1 / G.dx)) + ((1 / G.dy) * (1 / G.dy))) ** 0.5))
            G.mode = '2D TMz'
            G.pmlthickness['z0'] = 0
            G.pmlthickness['zmax'] = 0
    else:
        if dt>1 / (c * ((((1 / G.dx) * (1 / G.dx)) + ((1 / G.dy) * (1 / G.dy))) ** 0.5)):
            G.dt = 1 / (c * ((((1 / G.dx) * (1 / G.dx)) + ((1 / G.dy) * (1 / G.dy))) ** 0.5))
            print(G.dt)
            print(dt)
            print('dt is too large')
        else:
            G.dt = dt
        if G.nz:
            G.mode = '2D TMz'
            G.pmlthickness['z0'] = 0
            G.pmlthickness['zmax'] = 0
    if total_step == None:
        ts = step
    else:
        ts = total_step
    if customsource==None:
        G.source = torch.zeros((ts, nsrc, 3), device=device)
        G.source[:, :, 0] = ((torch.arange(nsrc, device=device) * source_step[0] + source_location[0]).repeat(ts, 1))
        G.source[:, :, 1] = ((torch.arange(nsrc, device=device) * source_step[1] + source_location[1]).repeat(ts, 1))
        G.source[:, :, 2] = ((torch.arange(nsrc, device=device) * source_step[2] + source_location[2]).repeat(ts, 1))
        ss = torch.arange(G.source.size(0), device=device).view(-1, 1, 1) * (source_step.to(device))
        G.source = G.source + ss
        G.source = torch.round(G.source / G.dx).int().contiguous()
    else:
        G.source=customsource
    
    if customreceiver==None:
        G.receiver = torch.zeros((ts, nrx, 3), device=device)
        G.receiver[:, :, 0] = ((torch.arange(nrx, device=device) * receiver_step[0] + receiver_location[0]).repeat(ts, 1))
        G.receiver[:, :, 1] = ((torch.arange(nrx, device=device) * receiver_step[1] + receiver_location[1]).repeat(ts, 1))
        G.receiver[:, :, 2] = ((torch.arange(nrx, device=device) * receiver_step[2] + receiver_location[2]).repeat(ts, 1))
        rs = torch.arange(G.receiver.size(0), device=device).view(-1, 1, 1) * (receiver_step.to(device))
        G.receiver = G.receiver + rs
        G.receiver = torch.round(G.receiver / G.dx).int().contiguous()
        if Fixedreceiver:
            G.receiver[:, :, :] = G.receiver[0, :, :].unsqueeze(0)
    else:
        G.receiver=customreceiver

    if isinstance(time_windows, torch.Tensor)==False:
        time_windows=torch.tensor(time_windows, device=device)
    
    # time_windows_tensor = time_windows.clone().detach()
    G_dt_tensor = G.dt
    G.iterations = int(torch.ceil(time_windows / G_dt_tensor)) + 1
    G.pmlthick = pmlthick
    G.pmlthickness['x0'] = pmlthick
    G.pmlthickness['y0'] = pmlthick
    G.pmlthickness['xmax'] = pmlthick
    G.pmlthickness['ymax'] = pmlthick

    w = Waveform()
    w.type = source_apmlitudes[0]
    w.amp = torch.tensor(source_apmlitudes[1], dtype=torch.float)
    w.freq = torch.tensor(source_apmlitudes[2], dtype=torch.float)
    if len(source_apmlitudes) > 3:
        w.delay = source_apmlitudes[3]
    G.waveforms.append(w)
    G.timewindow = time_windows.clone().detach()
    h = HertzianDipole()

    h.polarisation = source[1]
    h.start = 0
    h.stop = G.timewindow
    startstop = ' '
    h.calculate_waveform_values(G)
    if G.freq != source_apmlitudes[2] and G.freq != None:
        apply_filter(h.waveformvalues_wholestep, 1 / dt, freq)
    G.hertziandipoles.append(h)
    if pmlthick > 0:
        G.cfs = [CFS()]
        build_pmls(G, er, se, mr)

    G.initialise_std_update_coeff_arrays()

    # plot_er_with_source(er, G.source)

    erexpanded = torch.zeros((G.nx + 1, G.ny + 1, G.nz + 1), device=G.device)
    erexpanded[:G.nx, :G.ny, :G.nz] = er
    seexpanded = torch.zeros((G.nx + 1, G.ny + 1, G.nz + 1), device=G.device)
    seexpanded[:G.nx, :G.ny, :G.nz] = se
    mrexpanded = torch.zeros((G.nx + 1, G.ny + 1, G.nz + 1), device=G.device)
    mrexpanded[:G.nx, :G.ny, :G.nz] = mr

    libuc.Ucget(
        ctypes.cast(erexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(seexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(mrexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsE0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsE1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsH0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsH1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(G.updatecoeffsH4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        G.nx + 1,
        G.ny + 1,
        G.nz + 1,
        G.dt,
        G.dx
    )
    del erexpanded, seexpanded, mrexpanded
    G.updatecoeffsE0 = G.updatecoeffsE0.unsqueeze(0).repeat(G.step, 1, 1, 1)
    G.updatecoeffsE1 = G.updatecoeffsE1.unsqueeze(0).repeat(G.step, 1, 1, 1)
    G.updatecoeffsE4 = G.updatecoeffsE4.unsqueeze(0).repeat(G.step, 1, 1, 1)
    G.updatecoeffsH0 = G.updatecoeffsH0.unsqueeze(0).repeat(G.step, 1, 1, 1)
    G.updatecoeffsH1 = G.updatecoeffsH1.unsqueeze(0).repeat(G.step, 1, 1, 1)
    G.updatecoeffsH4 = G.updatecoeffsH4.unsqueeze(0).repeat(G.step, 1, 1, 1)
    torch.cuda.synchronize()
    G.gpu_initialise_arrays()
    for pml in G.pmls:
        pml.gpu_initialise_arrays()
        pml.update_electric_gpu = ('order' + str(len(pml.CFS)) + '_' + pml.direction)
        pml.update_magnetic_gpu = ('order' + str(len(pml.CFS)) + '_' + pml.direction)
    return G




class GPRFWI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G, er, se, mr):
        ctx.save_for_backward(er, se, mr)
        ctx.G = G
        if G.nseg == None:
            rxs_gpu, Ez ,Glast= Forward(G)
        else:
            rxs_gpu, Ez ,Glast= Forward(G)
        ctx.Ez = Ez
        ctx.Glast = Glast
        if torch.isnan(rxs_gpu).any():
            print("⚠️ rxs_gpu contains NaN!")
        return rxs_gpu[:, 2, :, :]

    @staticmethod
    def backward(ctx, grad_output):
        if torch.isnan(grad_output).any():
            print("⚠️ grad_output contains NaN!")
        er, se, mr = ctx.saved_tensors
        G = ctx.G
        Ez = ctx.Ez

        # if G.total_step != None:
        #     grad_er2 = torch.zeros((G.nx, G.ny, G.nz), device="cuda")
        #     grad_se2 = torch.zeros((G.nx, G.ny, G.nz), device="cuda")
        #     H = backinitialization(G, er, se, mr, erexpanded, seexpanded, mrexpanded)
        #     for i in range(int(H.total_step / H.step)):
        #         grad_outputi = grad_output[i * H.step:(i + 1) * H.step, :, :]
        #         grad_er3, grad_se3 = Backward(H, Ez, grad_outputi, i)
        #         grad_er2 += grad_er3
        #         grad_se2 += grad_se3
        # else:


        H = backinitialization(G, er, se, mr)
        grad_er2, grad_se2 = Backward(H, Ez, grad_output,er.requires_grad,se.requires_grad)

        
        if G.tv:
            epoch_start = time.time()
            if er.requires_grad:
                uer = total_variation(er.clone())
                λ = 0.01
                guer=2*(er-uer)
                grad_er2 += λ*guer  
                if torch.isnan(grad_er2).any():
                    print("⚠️ epsilon.grad contains NaN!")
                
            if se.requires_grad:
                use = total_variation(se.clone())
                λ = 0.01
                guse=2*(se-use)
                grad_se2 += λ*guse  
                if torch.isnan(grad_se2).any():
                    print("⚠️ sigma.grad contains NaN!")
            print(f"Total variation regularization took {time.time() - epoch_start:.4f} seconds.")


        #fengdeshan d
        # layer
        # if er.requires_grad:
        #     grad_er2[:H.pmlthick+3, :, :] = 0
        # if se.requires_grad:
        #     grad_se2[:H.pmlthick+3, :, :] = 0
        return None, grad_er2, grad_se2, None, None, None, None


def backinitialization(G, er, se, mr):
    global H
    H = FDTDGrid()
    H.step = G.step
    if G.total_step != 0:
        H.total_step = G.total_step

    H.pmlformulation = G.pmlformulation
    H.device = G.device
    H.dx = G.dx
    H.dy = G.dx
    H.dz = G.dx
    H.nx = G.nx
    H.ny = G.ny
    H.nz = G.nz
    H.nrx = G.nsrc
    H.nsrc = G.nrx
    H.dt = G.dt
    H.mode = G.mode
    H.pmlthickness['z0'] = G.pmlthickness['z0']
    H.pmlthickness['zmax'] = G.pmlthickness['zmax']
    H.receiver = G.receiver.contiguous()
    H.iterations = G.iterations
    H.pmlthick = G.pmlthick
    H.pmlthickness['x0'] = G.pmlthickness['x0']
    H.pmlthickness['y0'] = G.pmlthickness['y0']
    H.pmlthickness['xmax'] = G.pmlthickness['xmax']
    H.pmlthickness['ymax'] = G.pmlthickness['ymax']
    H.source = G.receiver

    del G

    if H.pmlthick > 0:
        H.cfs = [CFS()]
        build_pmls(H, er, se, mr)

    H.initialise_std_update_coeff_arrays()
    erexpanded = torch.zeros((H.nx + 1, H.ny + 1, H.nz + 1), device=H.device)
    erexpanded[:H.nx, :H.ny, :H.nz] = er
    seexpanded = torch.zeros((H.nx + 1, H.ny + 1, H.nz + 1), device=H.device)
    seexpanded[:H.nx, :H.ny, :H.nz] = se
    mrexpanded = torch.zeros((H.nx + 1, H.ny + 1, H.nz + 1), device=H.device)
    mrexpanded[:H.nx, :H.ny, :H.nz] = mr
    libuc.Ucgeta(
        ctypes.cast(erexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(seexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(mrexpanded.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsE0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsE1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsH0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsH1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(H.updatecoeffsH4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        H.nx + 1,
        H.ny + 1,
        H.nz + 1,
        H.dt,
        H.dx
    )
    del erexpanded, seexpanded, mrexpanded
    H.updatecoeffsE0 = H.updatecoeffsE0.unsqueeze(0).repeat(H.step, 1, 1, 1)
    H.updatecoeffsE1 = H.updatecoeffsE1.unsqueeze(0).repeat(H.step, 1, 1, 1)
    H.updatecoeffsE4 = H.updatecoeffsE4.unsqueeze(0).repeat(H.step, 1, 1, 1)
    H.updatecoeffsH0 = H.updatecoeffsH0.unsqueeze(0).repeat(H.step, 1, 1, 1)
    H.updatecoeffsH1 = H.updatecoeffsH1.unsqueeze(0).repeat(H.step, 1, 1, 1)
    H.updatecoeffsH4 = H.updatecoeffsH4.unsqueeze(0).repeat(H.step, 1, 1, 1)

    H.gpu_initialise_arrays()
    for pml in H.pmls:
        pml.gpu_initialise_arrays()
        pml.update_electric_gpu = ('order' + str(len(pml.CFS)) + '_' + pml.direction)
        pml.update_magnetic_gpu = ('order' + str(len(pml.CFS)) + '_' + pml.direction)
    return H


def Backward(H, Ez, grad_outputi,errequires_grad,serequires_grad,i=None):
    if errequires_grad:
        grad_er = torch.zeros((H.nx, H.ny, H.nz), device=H.device)
    if serequires_grad:
        grad_se = torch.zeros((H.nx, H.ny, H.nz), device=H.device)
    
    if i != None:
        H.Ex_gpu.zero_()
        H.Ey_gpu.zero_()
        H.Ez_gpu.zero_()
        H.Hx_gpu.zero_()
        H.Hy_gpu.zero_()
        H.Hz_gpu.zero_()
        sourcei = H.source[i * H.step:(i + 1) * H.step, :, :]
        srcwaveforms = grad_outputi.to(H.device)

        srcinfo1_ptr = sourcei.data_ptr()
        Ezi = Ez[:, i * H.step:(i + 1) * H.step, :, :, :]
    else:
        srcwaveforms = grad_outputi.to(H.device)
        srcinfo1_ptr = H.source.data_ptr()
        Ezi = Ez
        del Ez

    polar = 2
    srcinfo2_gpu = H.dz
    

    for iteration in range(H.iterations - 1, 0, -1):
        # print("-----"+str(iteration))
        if torch.isnan(H.Ez_gpu).any():
            print("⚠️ ez contains NaN!")
            break
        if torch.isnan(H.Hx_gpu).any():
            print("⚠️ hx contains NaN!")

        libback.back_source(
            H.step,
            iteration,
            H.dx, H.dy, H.dz,
            ctypes.cast(srcinfo1_ptr, ctypes.POINTER(ctypes.c_int)),
            srcinfo2_gpu,
            ctypes.cast(srcwaveforms.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            H.nx + 1,
            H.ny + 1,
            H.nz + 1,
            H.nsrc,
            polar,
            H.iterations
        )

        lib.e_fields_updates(
            ctypes.cast(H.updatecoeffsE0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.updatecoeffsE1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            H.step,
            H.nx + 1,  # a
            H.ny + 1,  # b
            H.nz + 1)  # c
        if torch.isnan(H.Ez_gpu).any():
            print("⚠️ ez contains NaN!")
        for pml in H.pmls:
            funcphe = getattr(libpe, pml.update_electric_gpu)
            funcphe.argtypes = libpe.argtypes
            funcphe.restype = libpe.restype

            EPhi1_gpu_ptr = pml.EPhi1_gpu.data_ptr()
            EPhi2_gpu_ptr = pml.EPhi2_gpu.data_ptr()
            ERA_gpu_ptr = pml.ERA_gpu.data_ptr()
            ERB_gpu_ptr = pml.ERB_gpu.data_ptr()
            ERE_gpu_ptr = pml.ERE_gpu.data_ptr()
            ERF_gpu_ptr = pml.ERF_gpu.data_ptr()

            funcphe(pml.xs, pml.xf, pml.ys,
                    pml.yf, pml.zs, pml.zf,
                    pml.EPhi1_gpu.shape[1], pml.EPhi1_gpu.shape[2], pml.EPhi1_gpu.shape[3],
                    pml.EPhi2_gpu.shape[1], pml.EPhi2_gpu.shape[2], pml.EPhi2_gpu.shape[3],
                    pml.thickness,
                    ctypes.cast(H.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(EPhi1_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(EPhi2_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERA_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERB_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERE_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(ERF_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    pml.d, ctypes.cast(H.updatecoeffsE4.data_ptr(), ctypes.POINTER(ctypes.c_float)), H.nx + 1, H.ny + 1, H.nz + 1, H.step)
        lib.h_fields_updates(
            ctypes.cast(H.updatecoeffsH0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.updatecoeffsH1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(H.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            H.step,
            H.nx + 1,  # a
            H.ny + 1,  # b
            H.nz + 1)

        for pml in H.pmls:
            funcphh = getattr(libph, pml.update_magnetic_gpu)
            funcphh.argtypes = libph.argtypes
            funcphh.restype = libph.restype

            HPhi1_gpu_ptr = pml.HPhi1_gpu.data_ptr()
            HPhi2_gpu_ptr = pml.HPhi2_gpu.data_ptr()
            HRA_gpu_ptr = pml.HRA_gpu.data_ptr()
            HRB_gpu_ptr = pml.HRB_gpu.data_ptr()
            HRE_gpu_ptr = pml.HRE_gpu.data_ptr()
            HRF_gpu_ptr = pml.HRF_gpu.data_ptr()

            funcphh(pml.xs, pml.xf, pml.ys,
                    pml.yf, pml.zs, pml.zf,
                    pml.HPhi1_gpu.shape[1], pml.HPhi1_gpu.shape[2], pml.HPhi1_gpu.shape[3],
                    pml.HPhi2_gpu.shape[1], pml.HPhi2_gpu.shape[2], pml.HPhi2_gpu.shape[3],
                    pml.thickness,
                    ctypes.cast(H.Ex_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Ey_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Ez_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hx_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hy_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(H.Hz_gpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HPhi1_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HPhi2_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRA_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRB_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRE_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(HRF_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
                    pml.d, ctypes.cast(H.updatecoeffsH4.data_ptr(), ctypes.POINTER(ctypes.c_float)), H.nx + 1, H.ny + 1, H.nz + 1, H.step)
        if torch.isnan(H.Hx_gpu).any():
            print("⚠️ hx contains NaN!")

        if errequires_grad:
            grad_er += torch.sum(
            (Ezi[iteration, :, :, :, :] - Ezi[iteration - 1, :, :, :, :]) * H.Ez_gpu[:, :-1, :-1, :-1],
            dim=0)
        if serequires_grad:
            grad_se += torch.sum(Ezi[iteration, :, :, :, :] * H.Ez_gpu[:, :-1, :-1, :-1]*H.dt, dim=0)

    if errequires_grad:
        if serequires_grad:
            return grad_er/grad_er.norm(), grad_se/grad_se.norm()
        else:  
            return grad_er/grad_er.norm(),None
    else:
        return None,grad_se/grad_se.norm()
    
