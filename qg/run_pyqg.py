import numpy as np
import pyqg
import xarray as xr
import pickle
import json
import argparse
import os
import shutil

parser = argparse.ArgumentParser(prog='2layerqg')
parser.add_argument('--config')
args = parser.parse_args()

with open(args.config, 'rb') as f:
    file_namelist = f.read().decode('utf-8')
namelist = json.loads(file_namelist)


namelist_input = namelist['input']
namelist_snapshot = namelist['snapshot']
namelist_output = namelist['output']


output_dir = namelist_output['output_dir']
exper_name = namelist_output['exper_name']
if not os.path.exists(os.path.join(output_dir, exper_name)):
    os.makedirs(os.path.join(output_dir, exper_name))


magnitude = namelist['forcing']['magnitude']
def q_parameterization(qg_model):
    
    # domain: [0, qg_model.L] [0, qg_model.W]
    # x coordinate : qg_model.x
    # y coordinate : qg_model.y
    # cell center grid (dx/2, dx/2+dx, ...)
    # time: qg_model.t
    
    W = qg_model.W
    
    f1 = -magnitude * np.sin(2*np.pi*qg_model.y/W)
    f2 =  magnitude * np.sin(2*np.pi*qg_model.y/W)
    
    return np.stack( (f1, f2) )


m = pyqg.QGModel(**namelist_input, q_parameterization=q_parameterization)
for _ in m.run_with_snapshots(**namelist_snapshot):
    ds = m.to_dataset()
    f_nc = os.path.join(output_dir, exper_name, exper_name+'.'+str(ds.attrs['pyqg:tc'])+'.nc')
    ds.to_netcdf(f_nc, 'w', engine="h5netcdf", invalid_netcdf=True) 
    f_pkl = os.path.join(output_dir, exper_name, exper_name+'.'+str(ds.attrs['pyqg:tc'])+'.pkl')
    m_diag = {key: value.copy() for key, value in m.diagnostics.items()}
    for key in m_diag.keys():
        m_diag[key].pop('function')
    with open(f_pkl, 'wb') as f:
        pickle.dump(m_diag, f, protocol=2)

shutil.copyfile(args.namelist, os.path.join(output_dir, exper_name, os.path.basename(args.namelist)))
