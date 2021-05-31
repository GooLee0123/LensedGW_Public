import numpy as np
from lensinggw.utils.utils import param_processing

np.random.seed(2)

def inject_background(lens_model, zL, zS, Macro_ra, Macro_dec):
    arc_to_rad = 4.84814e-6
    size = 600
    
    delta_ra = Macro_ra*1e-5
    delta_dec = Macro_dec*1e-5

    lower_ra = Macro_ra - delta_ra
    upper_ra = Macro_ra + delta_ra
    lower_dec = Macro_dec - delta_dec
    upper_dec = Macro_dec + delta_dec

    ras = np.random.uniform(lower_ra, upper_ra, size)
    decs = np.random.uniform(lower_dec, upper_dec, size)

    masses = np.random.uniform(100., 200., size)

    lens_list = []
    kwargs_list = []
    for i in range(size):
        lens_list.append(lens_model)
        thetaE = param_processing(zL, zS, masses[i])
        x, y = ras[i], decs[i]
        kwargs = {'center_x': x, 'center_y': y, 'theta_E': thetaE}
        kwargs_list.append(kwargs)

    return lens_list, kwargs_list
