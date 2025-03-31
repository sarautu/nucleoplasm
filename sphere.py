"""
To run using e.g. 128 processes:
    $ mpiexec -n 128 python3 sphere.py
    $ mpiexec -n 128 python3 sphere.py 24 48 0
    $ mpiexec -n 128 python3 sphere.py 24 48 0 --restart 5
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

# Parameters
R = 1 
Nphi, Ntheta, Nr = 96, 64, 128
dealias = 3/2
timestepper = d3.RK443
dtype = np.float64
mesh = None

# Model paramaters
Eta = 0.01 # Viscosity ratio
Gamma = 25 # Drag coefficient 
RhoBar = 100 # Crosslinking density

# regularization momentum diffusion
Epsilon = 0.05

# Initial time-step & stop-time
int_timestep = 5e-5
stop_sim_time = 0.5 + int_timestep 

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
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=R, dealias=dealias, dtype=dtype)
S2_basis = basis.S2_basis()

# Fields
v = dist.VectorField(coords, name='v', bases=basis)
tau_v = dist.VectorField(coords, name='tau_v', bases=S2_basis)

p = dist.Field(name='p', bases=basis)
tau_p = dist.Field(name='tau_p')

Q = dist.TensorField((coords, coords), name='Q', bases=basis)
tau_Q = dist.TensorField((coords, coords), name='tau_Q', bases=S2_basis)

nE = dist.Field(name='nE', bases=basis)
nH = dist.Field(name='nH', bases=basis)

uE = dist.VectorField(coords, name='uE', bases=basis)
uH = dist.VectorField(coords, name='uH', bases=basis)

lapE = dist.VectorField(coords, name='lapE', bases=basis)
lapH = dist.VectorField(coords, name='lapH', bases=basis)

tau_nE = dist.Field(name='tau_nE', bases=S2_basis)
tau_nH = dist.Field(name='tau_nH', bases=S2_basis)

tau_uE = dist.VectorField(coords, name='tau_uE', bases=S2_basis)
tau_uH = dist.VectorField(coords, name='tau_uH', bases=S2_basis)

# Substitutions
phi, theta, r = dist.local_grids(basis)
lift = lambda A: d3.Lift(A, basis, -1)

rrad = dist.VectorField(coords, name='rrad', bases=basis.radial_basis)
rrad['g'][2] = r

ephi = dist.VectorField(coords, name='ephi')
ephi['g'][0] = 1 #azimuthal unit vector

ethe = dist.VectorField(coords, name='ethe')
ethe['g'][1] = 1 #latitute unit vector

erad = dist.VectorField(coords, name='erad')
erad['g'][2] = 1 #radial unit vector


Q_phi_phi = d3.dot(d3.dot(Q, ephi), ephi)
Q_the_the = d3.dot(d3.dot(Q, ethe), ethe)
Q_rad_rad = d3.dot(d3.dot(Q, erad), erad)

Q_the_rad = d3.dot(d3.dot(Q, ethe), erad)
Q_rad_the = d3.dot(d3.dot(Q, erad), ethe)

Q_phi_the = d3.dot(d3.dot(Q, ephi), ethe)
Q_the_phi = d3.dot(d3.dot(Q, ethe), ephi)

Q_phi_rad = d3.dot(d3.dot(Q, ephi), erad)
Q_rad_phi = d3.dot(d3.dot(Q, erad), ephi)

T = dist.TensorField((coords, coords), name='T', bases=S2_basis)
T['g'] = 0
T['g'][0,0] = 1/2 # tangential anchoring  
T['g'][1,1] = 1/2 # tangential anchoring  
T['g'][2,2] = 0   # tangential anchoring

E = (d3.grad(v) + d3.trans(d3.grad(v)))/2
eH = (d3.grad(uH) + d3.trans(d3.grad(uH)))/2
eE = (d3.grad(uE) + d3.trans(d3.grad(uE)))/2

nE0 = 1.25*nE_bar
nE1 = nE - nE0

nH0_values = {10: 180, 20: 170, 30: 160, 40: 150}
default_nH0 = 150 
nH0 = nH0_values.get(nH_bar, default_nH0)
nH1 = nH - nH0
    
nH00 = 1.1*nH0
nH11 = nH - nH00

nT = nE + nH

upper= lambda u, A: d3.dot(u, d3.grad(A)) - d3.dot(d3.trans(d3.grad(u)),A) - d3.dot(A,d3.grad(u))

# Problem
if Alpha == 0:
    problem = d3.IVP([p, v, nE, nH, uE, uH, tau_p, tau_v, tau_nE, tau_nH, tau_uE, tau_uH], namespace=locals())
    problem.add_equation("lap(v) - grad(p) + lift(tau_v) - Gamma*nE0*(v-uE) - Gamma*nH0*(v-uH) = Gamma*nE1*(v-uE) + Gamma*nH1*(v-uH)")
else:
    problem = d3.IVP([p, v,  Q, nE, nH, uE, uH, tau_p, tau_v, tau_nE, tau_nH, tau_uE, tau_uH, tau_Q], namespace=locals())
    problem.add_equation("lap(v) - grad(p) + lift(tau_v) - Alpha*div(nE0*Q) - Gamma*nE0*(v-uE) - Gamma*nH0*(v-uH) = Gamma*nE1*(v-uE) + Gamma*nH1*(v-uH) + Alpha*div(nE1*Q)")
    problem.add_equation("dt(Q) - lap(Q) + lift(tau_Q) = -upper(v,Q) - 2*Q*trace(dot(Q,E))")
    problem.add_equation("dot(rrad, grad(Q))(r=1) = 0") # no flux
    #problem.add_equation("Q(r=1) = T") # tangential anchoring
    
problem.add_equation("div(v) + tau_p = 0")

problem.add_equation("dt(nE) - lap(nE) + lift(tau_nE) + div(nE0*uE) = -div(nE1*uE)")
problem.add_equation("dt(nH) - lap(nH) + lift(tau_nH) + div(nH0*uH) = -div(nH1*uH)")

problem.add_equation("2*(Epsilon + Eta*nH00)*div(eH) + lift(tau_uH) - Gamma*nH0*(uH-v) + RhoBar*grad(nH) - grad(nT*nH0)/2 = grad(nT*nH1)/2 + Gamma*nH1*(uH-v) - 2*Eta*div(nH11*eH)")
problem.add_equation("2*(Epsilon + Eta*nE0)*div(eE) + lift(tau_uE) - Gamma*nE0*(uE-v) - grad(nT*nE0)/2 = grad(nT*nE1)/2 + Gamma*nE1*(uE-v) - 2*Eta*div(nE1*eE)")

problem.add_equation("radial(grad(nE)(r=1)) = 0")   # No flux
problem.add_equation("radial(grad(nH)(r=1)) = 0")   # No flux

problem.add_equation("uE(r=1) = 0")  # No slip
problem.add_equation("uH(r=1) = 0")  # No slip

problem.add_equation("v(r=1) = 0")  # No slip
problem.add_equation("integ(p) = 0") 

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Apply initial conditions if not restarting from a checkpoint
if not restart:
    Q['g'] = 0
    Q['g'][2,2] = 1/3
    Q['g'][1,1] = 1/3 + 0.01*np.cos(3*phi)*np.sin(5*theta)*(r-1)**2
    Q['g'][0,0] = 1/3 - 0.01*np.cos(3*phi)*np.sin(7*theta)*(r-1)**2
    nH.fill_random('g', seed=50, distribution='normal', scale=1.5) # Random
    nH.low_pass_filter(scales=0.25)
    nE.fill_random('g', seed=20, distribution='normal', scale=1.5) # Random
    nE.low_pass_filter(scales=0.25)
    nE['g'] += nE_bar
    nH['g'] += nH_bar
    file_handler_mode = 'overwrite'
else:
    # Load state from checkpoint if restarting
    write, initial_timestep = solver.load_state(checkpoint_no)
    file_handler_mode = 'append'

# Setup output handlers for fields, flows, and scalars
checkpoints = solver.evaluator.add_file_handler(checkpoints, sim_dt=0.05, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

fields = solver.evaluator.add_file_handler(output_d, sim_dt=0.001, max_writes=50, mode=file_handler_mode)
fields.add_task(nE, name='nE')
fields.add_task(nH, name='nH')

if Alpha != 0:
    order = solver.evaluator.add_file_handler(output_o, sim_dt=0.001, max_writes=50, mode=file_handler_mode)
    order.add_task(Q_the_rad, name='Q_the_rad')
    order.add_task(Q_the_the, name='Q_the_the')
    order.add_task(Q_phi_phi, name='Q_phi_phi')
    order.add_task(Q_phi_the, name='Q_phi_the')
    order.add_task(Q_phi_rad, name='Q_phi_rad')
    
flows = solver.evaluator.add_file_handler(output_f, sim_dt=0.005, max_writes=10, mode=file_handler_mode)
flows.add_task(v, name='v')
flows.add_task(uE, name='uE')
flows.add_task(uH, name='uH')

# CFL
min_step_values = {0: 2.0e-4, 100: 1.0e-4, 150: 1.0e-4, 200: 1.0e-4, 250: 1.0e-4}
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
            Q['c'] = Q_sym.evaluate()['c']
            Q['g'][0,0] = 1 - Q['g'][1,1] - Q['g'][2,2]
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
