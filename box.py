"""
To run using e.g. 128 processes:
    $ mpiexec -n 128 python3 box.py
    $ mpiexec -n 128 python3 box.py 24 24 0
    $ mpiexec -n 128 python3 box.py 24 24 0 --restart
    $ mpiexec -n 128 python3 box.py 24 24 0 --restart 5
"""
import sys
import numpy as np
import dedalus.public as d3
import logging
import os
import glob
import re
logger = logging.getLogger(__name__)

# Parse command line argument to check if restarting from a checkpoint
restart = (len(sys.argv) > 4 and sys.argv[4] == '--restart')

# Set domain dimensions and resolution
L = 2
Lx, Ly, Lz = L, L, L
Nx, Ny, Nz = 128, 128, 128
dealias = 3/2
timestepper = d3.RK443
dtype = np.float64

# Model paramaters
Eta = 0.01 # Viscosity ratio
Gamma = 15 #25 # Drag coefficient 
RhoBar = 100 # Crosslinking density

# regularization momentum diffusion
Epsilon = 0.05

# Initial time-step & stop-time
int_timestep = 5e-5
stop_sim_time = 1 + int_timestep 

# Command line arguments to set initial conditions and output directories
if len(sys.argv) > 1:
    nE_bar = float(sys.argv[1])
    nH_bar = float(sys.argv[2])
    Alpha  = float(sys.argv[3])
    # Output directory names based on command line arguments
    output_d = "fields_nE_{:02d}_nH_{:02d}_Alpha_{:03d}".format(round(nE_bar), round(nH_bar), round(Alpha))
    output_f = "flows_nE_{:02d}_nH_{:02d}_Alpha_{:03d}".format(round(nE_bar), round(nH_bar), round(Alpha))
    output_o = "order_nE_{:02d}_nH_{:02d}_Alpha_{:03d}".format(round(nE_bar), round(nH_bar), round(Alpha))
    checkpoints = "checkpoints_nE_{:02d}_nH_{:02d}_Alpha_{:03d}".format(round(nE_bar), round(nH_bar), round(Alpha))
    
    if restart:
        # If a checkpoint number is provided as a command line argument, use that file
        if len(sys.argv) > 5:
            manual_checkpoint = str(sys.argv[5])
            checkpoint_no = os.path.join(checkpoints, checkpoints + "_s" + manual_checkpoint + ".h5")
        else:
            # Otherwise, automatically find the checkpoint file with the highest number
            pattern = os.path.join(checkpoints, checkpoints + "_s*.h5")
            checkpoint_files = glob.glob(pattern)
            
            if checkpoint_files:
                # Extract the integer following '_s' in the filename
                def extract_checkpoint_number(filename):
                    match = re.search(r'_s(\d+)\.h5$', filename)
                    return int(match.group(1)) if match else -1
                
                last_checkpoint_file = max(checkpoint_files, key=extract_checkpoint_number)
                checkpoint_no = last_checkpoint_file
            else:
                logger.error("No checkpoint file found in %s", checkpoints)
                restart = False
else:
    # Default parameters if no command line arguments are provided
    nE_bar = 20
    nH_bar = 20
    Alpha  = 0
    output_d = "fields"
    output_f = "flows"
    output_o = "order"
    checkpoints = "checkpoints"

# Log the start of the simulation
logger.info('Running: nE = ' + str(round(nE_bar)) + ',  nH = ' + str(round(nH_bar)) + ',  Alpha = ' + str(round(Alpha)))
if restart:
    logger.info('Restarting from file: ' + checkpoint_no)

# Setup domain and field bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Define simulation fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
v = dist.VectorField(coords, name='v', bases=(xbasis,ybasis,zbasis))
nE = dist.Field(name='nE', bases=(xbasis,ybasis,zbasis))
nH = dist.Field(name='nH', bases=(xbasis,ybasis,zbasis))
uE = dist.VectorField(coords, name='uE', bases=(xbasis,ybasis,zbasis))
uH = dist.VectorField(coords, name='uH', bases=(xbasis,ybasis,zbasis))
Q = dist.TensorField((coords, coords), name='Q', bases=(xbasis,ybasis,zbasis))

scalar_order = dist.Field(name='scalar_order', bases=(xbasis, ybasis, zbasis))
director = dist.VectorField(coords, name='director', bases=(xbasis, ybasis, zbasis))

# Tau fields
tau_p = dist.Field(name='tau_p')
tau_v = dist.VectorField(coords, name='tau_v')
tau_nE = dist.Field(name='tau_nE')
tau_nH = dist.Field(name='tau_nH')
tau_uE = dist.VectorField(coords, name='tau_uE')
tau_uH = dist.VectorField(coords, name='tau_uH')

# Substitutions
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

Q_xx = d3.dot(d3.dot(Q, ex), ex)
Q_yy = d3.dot(d3.dot(Q, ey), ey)
Q_xy = d3.dot(d3.dot(Q, ex), ey)
Q_xz = d3.dot(d3.dot(Q, ex), ez)
Q_yz = d3.dot(d3.dot(Q, ey), ez)

E = (d3.grad(v) + d3.trans(d3.grad(v)))/2

eH = (d3.grad(uH) + d3.trans(d3.grad(uH)))/2
eE = (d3.grad(uE) + d3.trans(d3.grad(uE)))/2

nE0 = 1.25*nE_bar
nE1 = nE - nE0

nH0_values = {10: 190, 20: 160, 30: 160, 40: 160, 60: 160}
default_nH0 = 190 
nH0 = nH0_values.get(nH_bar, default_nH0)
nH1 = nH - nH0
    
nH00 = 1.1*nH0
nH11 = nH - nH00

nT = nE + nH

upper= lambda u, A: d3.dot(u, d3.grad(A)) - d3.dot(d3.trans(d3.grad(u)),A) - d3.dot(A,d3.grad(u))

# Define the problem and equations
if Alpha == 0:
    problem = d3.IVP([p, v, nE, nH, uE, uH, tau_p, tau_v, tau_nE, tau_nH, tau_uE, tau_uH], namespace=locals())
    # Momentum equation for the incompressible solvent 
    problem.add_equation("lap(v) - grad(p) + tau_v - Gamma*nE0*(v-uE) - Gamma*nH0*(v-uH) = Gamma*nE1*(v-uE) + Gamma*nH1*(v-uH)")
else:
    problem = d3.IVP([p, v,  Q, nE, nH, uE, uH, tau_p, tau_v, tau_nE, tau_nH, tau_uE, tau_uH], namespace=locals())
    # Momentum equation for the incompressible solvent 
    problem.add_equation("lap(v) - grad(p) + tau_v - Alpha*div(nE0*Q) - Gamma*nE0*(v-uE) - Gamma*nH0*(v-uH) = Gamma*nE1*(v-uE) + Gamma*nH1*(v-uH) + Alpha*div(nE1*Q)")
    # Equation of the orientational order parameter 
    problem.add_equation("dt(Q) - lap(Q) = -upper(v,Q) - 2*Q*trace(dot(Q,E))")
    
problem.add_equation("div(v) + tau_p = 0") #Solvent incompressibility condition 

# Conservation equations for heterochromatin and euchromatin 
problem.add_equation("dt(nE) - lap(nE) + tau_nE + div(nE0*uE) = -div(nE1*uE)")
problem.add_equation("dt(nH) - lap(nH) + tau_nH + div(nH0*uH) = -div(nH1*uH)")

# Momentum equations for heterochromatin and euchromatin
problem.add_equation("2*(Epsilon + Eta*nH00)*div(eH) + tau_uH - Gamma*nH0*(uH-v) + RhoBar*grad(nH) - grad(nT*nH0)/2 = grad(nT*nH1)/2 + Gamma*nH1*(uH-v) - 2*Eta*div(nH11*eH)")
problem.add_equation("2*(Epsilon + Eta*nE0)*div(eE) + tau_uE - Gamma*nE0*(uE-v) - grad(nT*nE0)/2 = grad(nT*nE1)/2 + Gamma*nE1*(uE-v) - 2*Eta*div(nE1*eE)")

# Integral constraints for pressure, velocity, and density
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(v) = 0")

problem.add_equation("integ(uE) = 0")
problem.add_equation("integ(uH) = 0")

problem.add_equation("integ(nE) = nE_bar*Lx*Ly*Lz")
problem.add_equation("integ(nH) = nH_bar*Lx*Ly*Lz")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Apply initial conditions if not restarting from a checkpoint
if not restart:
    Q['g'] = 0
    Q['g'][1,1] = 1/3 + 0.01*(np.cos(3*2*np.pi*x/Lx) + np.cos(4*2*np.pi*y/Ly) + np.cos(5*2*np.pi*z/Lz))
    Q['g'][2,2] = 1/3 - 0.01*(np.cos(3*2*np.pi*x/Lx) + np.cos(4*2*np.pi*y/Ly) + np.cos(5*2*np.pi*z/Lz))
    Q['g'][0,0] = 1/3
    nH.fill_random('g', seed=50, distribution='normal', scale=1.5) # Random
    nH.low_pass_filter(scales=0.25)
    nE.fill_random('g', seed=20, distribution='normal', scale=1.5) # Random
    nE.low_pass_filter(scales=0.25)
    nE['g'] += nE_bar
    nH['g'] += nH_bar
    file_handler_mode = 'overwrite'
else:
    # Load state from checkpoint if restarting
    write, int_timestep = solver.load_state(checkpoint_no)
    file_handler_mode = 'append'

# Setup output handlers for fields and flows
checkpoints = solver.evaluator.add_file_handler(checkpoints, sim_dt=0.05, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

fields = solver.evaluator.add_file_handler(output_d, sim_dt=0.001, max_writes=50, mode=file_handler_mode)
fields.add_task(nE, name='nE')
fields.add_task(nH, name='nH')

if Alpha != 0:
    order = solver.evaluator.add_file_handler(output_o, sim_dt=0.001, max_writes=50, mode=file_handler_mode)
    order.add_task(scalar_order, name='order')
    order.add_task(director, name='director')

flows = solver.evaluator.add_file_handler(output_f, sim_dt=0.005, max_writes=10, mode=file_handler_mode)
flows.add_task(v, name='v')
flows.add_task(uE, name='uE')
flows.add_task(uH, name='uH')
    
# CFL
min_step_values = {0: 2.0e-4, 100: 1.0e-4, 250: 0.5e-4, 500: 0.25e-4}
default_min_step = 0.5e-4 
min_step = min_step_values.get(Alpha, default_min_step)

CFL = d3.CFL(solver, initial_dt=int_timestep, cadence=1, safety=1/2, max_change=1.5, min_change=0.5, max_dt=5e-4, min_dt=min_step, threshold=0)
CFL.add_velocity(v)
CFL.add_velocity(uE)
CFL.add_velocity(uH)

# Track flow properties globally for diagnostics
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(nH, name='rhoH')
flow.add_property(nE, name='rhoE')
flow.add_property(d3.dot(uH,uH), name='uH2')
flow.add_property(d3.dot(uE,uE), name='uE2')
flow.add_property(d3.dot(v,v), name='v2')

# Rounding CFL timestep
def rounding(n):
    if n == 0:
        return 0
    e = np.floor(np.log10(abs(n)))
    return np.sign(n) * round(abs(n) / 10**e, 1) * 10**e

# Main loop
if Alpha != 0:
    Q_sym = (Q + d3.transpose(Q)) / 2
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = rounding(CFL.compute_timestep())
        solver.step(timestep)
        if Alpha != 0:
            # Enforce tensor symmetry and tr(Q) = 1
            Q['c'] = Q_sym.evaluate()['c']
            Q['g'][0,0] = 1 - Q['g'][1,1] - Q['g'][2,2]
            
            # --- Compute maximum eigenvalue and eigenvectors ---
            temp = Q.copy()
            temp.change_scales(dealias)
            matrix_data = np.transpose(temp['g'], (2, 3, 4, 0, 1))
            eigvals, eigvecs = np.linalg.eigh(matrix_data)
            max_eig = eigvals[..., -1]
            max_eigvec = eigvecs[..., :, -1]
            
            scalar_order.change_scales(dealias)
            scalar_order['g'] = (3*max_eig-1)/2
            scalar_order.change_scales(1)
            
            director.change_scales(dealias)
            director['g'] = np.transpose(max_eigvec, (3, 0, 1, 2))
            director.change_scales(1)
            # -----------------------------------
        
        if (solver.iteration-1) % 20 == 0:
            max_v = np.sqrt(flow.max('v2'))
            max_uE = np.sqrt(flow.max('uE2'))
            max_uH = np.sqrt(flow.max('uH2'))
            max_rhoE = flow.max('rhoE')
            max_rhoH = flow.max('rhoH')
            logger.info("Iteration=%i, Time=%.4f, Step=%.1e, Max: nE=%.1f, nH=%.1f, v=%.2f, uE=%.2f, uH=%.2f"%(solver.iteration, solver.sim_time, timestep, max_rhoE, max_rhoH, max_v, max_uE, max_uH))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
