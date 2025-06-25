import ctypes
import os
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as m0
import subprocess
import os


from pathlib import Path

lib_dir = Path(__file__).parent / 'lib'
so_files = ['back.so', 'fields_updates_gpu.so','sourcereceiver.so', 'uc.so', 'pml_updates_h.so', 'pml_updates_e.so']

if all((lib_dir / f).is_file() for f in so_files)==False:
    print('Compiling CUDA files...')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    make_command = ['make', '-C', current_dir]
    subprocess.run(make_command, check=True)


z0 = (m0 / e0) ** 0.5

lib_path = os.path.join(os.path.dirname(__file__), 'lib', 'fields_updates_gpu.so')
libsr_path = os.path.join(os.path.dirname(__file__), 'lib', 'sourcereceiver.so')
libuc_path = os.path.join(os.path.dirname(__file__), 'lib', 'uc.so')
libph_path= os.path.join(os.path.dirname(__file__), 'lib', 'pml_updates_h.so')
libpe_path = os.path.join(os.path.dirname(__file__), 'lib', 'pml_updates_e.so')
libback_path=os.path.join(os.path.dirname(__file__), 'lib', 'back.so')

lib = ctypes.cdll.LoadLibrary(lib_path)
libsr = ctypes.cdll.LoadLibrary(libsr_path)
libuc = ctypes.cdll.LoadLibrary(libuc_path)
libph = ctypes.cdll.LoadLibrary(libph_path)
libpe = ctypes.cdll.LoadLibrary(libpe_path)
libback = ctypes.cdll.LoadLibrary(libback_path)



libpe.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]
libpe.restype = None


libback.back_source.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
libback.back_source.restype = None



lib.e_fields_updates.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
lib.e_fields_updates.restype = None

lib.h_fields_updates.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
lib.h_fields_updates.restype = None


libsr.launch_store_outputs.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
libsr.launch_store_outputs.restype = None


libuc.Ucget.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float
]
libuc.Ucget.restype = None

libuc.Ucgeta.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float
    ]
libuc.Ucgeta.restype = None

libsr.launch_store_outputs.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
libsr.launch_store_outputs.restype = None

libsr.update_hertzian_dipole.argtypes = [
    ctypes.c_int,
    ctypes.c_int,  # iteration
    ctypes.c_float, ctypes.c_float, ctypes.c_float,  # dx, dy, dz
    ctypes.POINTER(ctypes.c_int),  # d_srcinfo1
    ctypes.c_float,  # d_srcinfo2
    ctypes.POINTER(ctypes.c_float),  # d_srcwaveforms
    ctypes.POINTER(ctypes.c_float),  # d_Ex
    ctypes.POINTER(ctypes.c_float),  # d_Ey
    ctypes.POINTER(ctypes.c_float),  # d_Ez
    ctypes.POINTER(ctypes.c_float),  # d_uE4
    ctypes.c_int,  # NY_SRCINFO
    ctypes.c_int,  # NY_SRCWAVES
    ctypes.c_int,  # NY_FIELDS
    ctypes.c_int,  # NZ_FIELDS
    ctypes.c_int,
    ctypes.c_int
]
libsr.update_hertzian_dipole.restype = None


libph.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libph.restype = None

__all__ = ['lib', 'libsr']
