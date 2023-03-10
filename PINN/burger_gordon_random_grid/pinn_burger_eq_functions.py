# Rather than carry a dictionary, the more elegant approach is to use a class, 
# where one of the classes is in its own file, and can be accessed from all the other files. 
# Ideally, the class would be a singleton so that I could instantiate it fram anywhere and 
# access the global variables. If an attribute of this class were changed, all instances would
# access the changed data. CAN THIS BE DONE IN PYTHON? In C++, I would use a global class. 
# In Julia, I would use a non-mutable struct with a single entry: the dictionary. 


import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
import torch
from torch import nn
from  typing import Callable
from functools import partial
from torch import tensor as tt
from DictX import GlobDct, DictX
import ipywidgets as widgets

#----------------------------------------------------------------------
def xrand(N, x0, x1):
    x = x0 + (x1-x0) * torch.rand(N).reshape(-1,1)
    x.requires_grad = True    # Crucial
    return x
#----------------------------------------------------------------------
def IC(N_IC, x0, xL, **kwargs) -> torch.Tensor:
    dct = GlobDct()
    # reshape(-1,1) implies that the batch size is all the points
    # x: torch.tensor of points 
    dct.x_IC = xrand(N_IC, x0, xL)
    # print("x min/max: ", torch.min(dct.x_IC), torch.max(dct.x_IC))
    dct.t_IC = torch.zeros(N_IC).reshape(-1,1) 
    # [[x1,t1], [x2,t2], ...]:  (Batch, 2)
    dct.pts_IC = torch.cat([dct.x_IC, dct.t_IC], dim=1)  # (x, t) pairs
    # One period of sine wave for any xL
    dct.u_IC = -torch.sin(2.*np.pi * dct.x_IC / xL).reshape(-1, 1)
    return dct.pts_IC, dct.u_IC

#----------------------------------------------------------------------
def BC(N_BC, x0, xL, T0, T1, u0, uL, **dct_):
    """ 
    x ranges from x0 to xL 
    Assume boundary condition constant in time
    t is an torch tensor of random times in the computation interval
    Note: **dct_ dereferences the dictionary, but I can still access the global dct
    This approach ensures a certain level of consistency for the function arguments.
    The disadvantage of this approch is that a change to the key might require an update to 
    all the functions using this dictionary. Nonetheless, the current approach is quite convenient. 
    """
    dct = GlobDct()
    dct.t_left  = xrand(N_BC, T0, T1)
    dct.t_right = xrand(N_BC, T0, T1)
    dct.u_l = u0.repeat(N_BC)  # fixed in time
    dct.u_r = uL.repeat(N_BC)
    dct.x_l = x0.repeat(N_BC).reshape(-1,1).requires_grad_(True)
    dct.x_r = xL.repeat(N_BC).reshape(-1,1).requires_grad_(True)
    dct.pts_left_BC = torch.cat([dct.x_l, dct.t_left], dim=1)
    dct.pts_right_BC = torch.cat([dct.x_r, dct.t_right], dim=1)
    return dct.pts_left_BC, dct.pts_right_BC, dct.u_l, dct.u_r

#----------------------------------------------------------------------
def interior(N, x0, xL, T0, T1, **kwargs):
    """
    Generate an array of size N with t values and 
    another with N values of x. 
    These points change every epoch. 
    If the argument 'is_random:Bool' is present, generate a tensor-product
    grid if True with N points in x and N points in T
    """
    dct = GlobDct()
    # how to update a dict A with another dict B
    # A.update(B)
    if "is_random" not in kwargs:
        dct.is_random = False
    else:
        dct.is_random = kwargs["is_random"]

    if dct.is_random:
        dct.x = xrand(N, x0, xL)
        dct.t = xrand(N, T0, T1)
    else:
        # Exclude boundary points [1:-1] and [1:]
        # Create N x N tensor grid
        x = tt(np.linspace(x0, xL, N)[1:-1]).float()
        t = tt(np.linspace(T0, T1, N)[1:]).float()
        x.requires_grad = True
        t.requires_grad = True
        if t.shape[0] > 200 or x.shape[0] >200:
            print("Shape too large!")
            return
        dct.x, dct.t = torch.meshgrid(x, t)
        # all the points in a tensor grid are flattened to the first index
        dct.x = dct.x.reshape(-1,1)
        dct.t = dct.t.reshape(-1,1)
        # x[:, 0]: (x)
        # t[:, 0]: (t=T0)
        # t[:, -1]: (t=T1)
        #print("dct.x[:,0]: ", dct.x[:,0]) # range of x
        #print("dct.t[:,0]: ", dct.t[:,0]) # 0.035
        #print("dct.t[:,-1]: ", dct.t[:,-1]) # 1

    # Concatenate x and t to form a tensor of (x,t) pairs
    dct.pts = torch.cat([dct.x, dct.t], dim=1)
    return dct.pts
#----------------------------------------------------------------------
class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation

    num_hidden: number of hiddent layers (in addition to input and output layers)
    dim_hidden: number of neurons in each hidden layer
    
    Input to the Neural Network is a collection of (x,t) points.
    Output of the network is the solution u. 
    I will need the x,t derivatives of the solution. 
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super(PINN, self).__init__()
        # input t into the network
        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
        # self.init_weights()

    def forward(self, t):
        # t must be of type Float
        t = t.reshape(-1, 2) 
        out = self.act(self.layer_in(t)) # ERROR? 
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # print("Initialize weights to zero.")
                # nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
#----------------------------------------------------------------------
def f(arr: torch.tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return arr


def df(f_out: torch.Tensor, f_in: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    # Loop over order to egt higher
    for _ in range(order):
        df_value = torch.autograd.grad(
            f_out,
            f_in,
            grad_outputs=torch.ones_like(f_out),
            create_graph=True,  # Necessary if a second derivative will be calculated? 
        )[0]

    return df_value

grad = torch.autograd.grad

def dudt(dep_var, t: torch.Tensor):
    return grad(dep_var, t, grad_outputs=torch.ones_like(dep_var), create_graph=True, retain_graph=True)[0]  # why should this be true?

def dudx_dudxx(dep_var, x: torch.Tensor):
    # I need ux to compute uxx, so return them both
    ux  = grad(dep_var, x, grad_outputs=torch.ones_like(dep_var), create_graph=True, retain_graph=True)[0]
    uxx = grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=False, retain_graph=True)[0]   # Why should this be True? 
    return ux, uxx
#--------------------------------------------------------------------------------------------
def draw_domain(N, x0, xL, T0, T1, pts, pts_IC, pts_left_BC, pts_right_BC, **kwargs):
    if kwargs['is_random'] == True:
        N = 5000
    else:
        N = 30
    @interact(ms_=(2, 60, 5), fontsize_=(10, 40), randomize_=False, N=N)
    def draw_physical_domain(ms_, fontsize_, randomize_, N, **dct_):
        dct = GlobDct();
        dct.N = N
        #if randomize_:
        if True:
            dct = GlobDct();
            #interior(dct.N, dct.x0, dct.xL, dct.T0, dct.T1)
            interior(**dct)
            #BC(dct.N_BC, dct.x0, dct.xL, dct.T0, dct.T1, dct.u0, dct.uL)
            BC(**dct)
            #IC(dct.N_IC, dct.x0, dct.xL)
            IC(**dct)
        plt.figure(figsize=(10, 10))
        pts1 = dct.pts.detach()  # interior points
        pts_IC1 = dct.pts_IC.detach().numpy()
        pts_left_BC1 = dct.pts_left_BC.detach()
        pts_right_BC1 = dct.pts_right_BC.detach()
        # create a scatter plot with a marker size of 1
        plt.scatter(pts1[:,0], pts1[:,1], s=ms_, color='c', alpha=1, label="Interior")
        plt.scatter(pts_IC1[:,0], pts_IC1[:,1], s=ms_, color='m', alpha=1, label="IC")
        plt.scatter(pts_left_BC1[:,0], pts_left_BC1[:,1], s=ms_, color='g', alpha=1, label="BC")
        plt.scatter(pts_right_BC1[:,0], pts_right_BC1[:,1], s=ms_, color='g', alpha=1, label="BC")
        x_offset = 0.05 * (xL - x0)
        t_offset = 0.05 * (T1 - T0)
        plt.xlim(x0-x_offset, xL+x_offset)
        plt.ylim(T0-t_offset, T1+t_offset)
        plt.grid(True, alpha=.5)
        plt.xlabel('x', fontsize=fontsize_)
        plt.ylabel('y', fontsize=fontsize_)
        plt.legend(fontsize=fontsize_)
        plt.title("Interior, Boundary, and Initial-Value points", fontsize=fontsize_*1.4)
#----------------------------------------------------------------------
def draw_initial_conditions():
    dct = GlobDct()
    @interact(ms_=(5, 60), fontsize_=(10, 40), alpha_=(0.,1.), randomize_=False, run=True)
    def initial_conditions(ms_, fontsize_, alpha_, randomize_, run, **dct_):
        # Access global dictionary
        if randomize_:
            IC(dct=dct, **dct)
        plt.figure(figsize=(10, 10))
        # t = dct.pts_IC.detach().numpy()[:,1]
        x = dct.pts_IC.detach().numpy()[:,0]
        u = dct.u_IC.detach().numpy().reshape(-1)

        # create a scatter plot with a marker size of 1
        plt.scatter(x, u, s=ms_, color='c', alpha=1.)

        argx = x.argsort()
        x = x[argx]
        u = u[argx]
        plt.plot(x, u, color='w', alpha=alpha_)

        x_offset = 0.05 * (dct.xL - dct.x0)
        umn, umx = u.min(), u.max()
        u_offset = 0.05 * (umx - umn)

        plt.xlim(dct.x0-x_offset, dct.xL+x_offset)
        plt.ylim(umn - u_offset, umx + u_offset)

        plt.grid(True, alpha=.5)
        plt.xlabel('x', fontsize=fontsize_)
        plt.ylabel('t', fontsize=fontsize_)
        plt.title("Interior, Boundary, and Initial-Value points", fontsize=fontsize_*1.4)
#---------------------------------------------------------------------
def compute_loss(nn_approximator: PINN) -> torch.float:
    dct = GlobDct()
    nu = dct.nu
    
    pts = interior(**dct)
    u = nn_approximator(pts)

    ux, uxx = dudx_dudxx(u, dct.x) #.reshape(-1,1)  # Note ux argument
    ut = dudt(u, dct.t)
    
    # Boundary condition (random t)
    BC(**dct)  # update dct
    unn_l_BC  = nn_approximator(dct.pts_left_BC)
    unn_r_BC = nn_approximator(dct.pts_right_BC)
    
    IC(**dct)  # updates dct
    unn_IC = nn_approximator(dct.pts_IC)
    
    # Points are unsorted
    # sine wave as expected, at every iteration
    plt.scatter(unwrap(dct.x_IC), unwrap(dct.u_IC))  
    
    # Loss functions
    interior_loss = (ut + u*ux - nu*uxx).pow(2).mean()
    IC_loss = (unn_IC - dct.u_IC).pow(2).mean()
    BC_loss = (unn_l_BC - dct.u_l).pow(2).mean() + \
              (unn_r_BC - dct.u_r).pow(2).mean()
    
    # Can add loss due to observations if we had any
    # Obs_loss = ...

    # Total loss
    final_loss =  interior_loss + IC_loss + BC_loss
    #    + Obs_loss

    return IC_loss, BC_loss, interior_loss, final_loss
#---------------------------------------------------------------------
def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.001,
    max_epochs: int = 10
    ) -> PINN:
    
    dct = GlobDct()
    dct.IC_losses = []
    dct.BC_losses = []
    dct.interior_losses = []
    dct.total_losses = []

    optimizer = torch.optim.AdamW(nn_approximator.parameters(), lr=learning_rate, weight_decay=0.005)
    skip = max_epochs // 10
    #optimizer = torch.optim.SGD(nn_approximator.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

        #t = torch.linspace(t_domain[0], t_domain[1], steps=nt, requires_grad=True)
        # Different set of t every epoch
        #t = torch.rand(nt, requires_grad=True) * (t_domain[1] - t_domain[0]) + t_domain[0]
        #t[0] = 0.0
        loss_fn = partial(compute_loss)

        try:
            optimizer.zero_grad()
            IC_loss, BC_loss, interior_loss, total_loss = loss_fn(nn_approximator)

            # if epoch % skip == 0:
                # print("Interior, IC, BC loss: ", 
                    # interior_loss.item(), IC_loss.item(), BC_loss.item())

            dct.total_losses.append(total_loss.detach().item())
            dct.BC_losses.append(BC_loss.detach().item())
            dct.IC_losses.append(IC_loss.detach().item())
            dct.interior_losses.append(interior_loss.detach().item())

            total_loss.backward()  # was not in orig
            optimizer.step()

        # If type ctrl-C on the keyboard, this loop is exited, and then the program continues
        except KeyboardInterrupt:
            break

    return nn_approximator
#----------------------------------------------------------------------
def plot_solution(nn_trained: PINN):
    fig, axes = plt.subplots(1, 3, figsize=(25,7))
    axes = axes.reshape(-1)
    ax = axes[0]
    ax.clear()
    
    dct = GlobDct()
    nu = dct["nu"] 
    x0 = tt(0.)
    x1 = xL = dct["xL"]
    u0 = dct["u0"] 
    uL = dct["uL"] 
    T0 = dct["T0"] 
    T1 = dct["T1"] 
    N = dct["N"] 
    N_IC = dct["N_IC"]
    N_BC = dct["N_BC"] 
    
    # Generate a solution on a regular grid
    
    N = 50
    x = torch.linspace(x0, x1, steps=N)
    # print("dcct: ", dct.keys())
    t = torch.linspace(T0, T1, steps=N)
    x, t = torch.meshgrid(x, t)
    
    x = x.reshape(-1,1)
    t = t.reshape(-1,1)
    pts = torch.cat([x,t], dim=1)
    sol = nn_trained(pts).detach().reshape(-1).detach()
    sol = sol.reshape(N,N).detach()
    t = t.reshape(N,N).detach()
    x = x.reshape(N,N).detach()
    
    # Compoute exact solution at 100 points as a curve
    exact_sol = torch.exp(-t.detach())
    
    for i in range(0,50):  # CHECK RANGE
         ax.plot(x[:,i], sol[:,i])
    ax.set_xlabel("x", fontsize=dct.fontsz)
    ax.set_ylabel("solution", fontsize=16)
    ax.set_title("u(x,:) at different times", fontsize=dct.fontsz)
    ax.grid(True)
            
    ax = axes[1]
    for i in range(0,50):  # CHECK RANGE
         ax.plot(t[i,:], sol[i,:])
    ax.set_xlabel("t", fontsize=dct.fontsz)
    ax.set_ylabel("solution")
    ax.set_title("u(:,t) at different locations", fontsize=dct.fontsz)
        
    #ax.scatter(t.detach().reshape(-1), exact_sol, label='exact')
    plt.legend()
    
    ax = axes[2]
    ax.clear()
    ax.plot(dct.total_losses, label="total")
    ax.plot(dct.interior_losses, label="interior")
    ax.plot(dct.IC_losses, label="ic")
    ax.plot(dct.BC_losses, label="bc")
    ax.set_title("loss10", fontsize=dct.fontsz)
    ax.set_xlabel("epoch")
    ax.set_ylabel("$log_{10} (loss)$")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(fontsize=dct.fontsz)
    plt.tight_layout()
    plt.show()
#----------------------------------------------------------------------
def plot_run():
    """
    height = {'orientation':'vertical', "layout":widgets.layout(height='175px', margin='10px')}
    width = {"layout":widgets.layout(width='175px', margin='20px')}
    nb_epochs = widgets.intslider(min=5, max=2005, step=200, value=50, description="#epochs", **height)
    nb_hid_layers = widgets.intslider(min=2. , max=5., description='#layers', **height)
    lr = widgets.floatslider(min=1.e-4 , max=1.e-1, description='lr', **height)
    pts_per_layer = widgets.intslider(min=10 , max=60, step=10, description='#pts in layer', **height)
    n = widgets.intslider(min=200 , max=2000, step=200, description='n', **height)
    n_ic = widgets.intslider(min=10 , max=111, step=10, description='n_ic', **height)
    n_bc = widgets.intslider(min=10 , max=211, step=10, description='n_bc', **height)
    fontsz = widgets.intslider(min=12 , max=48, step=4, description='font size', **height)
    t1 = widgets.floatslider(min=0.5 , max=3., step=0.5, value=1., description='max(t)', **height)
    """

    dct = globdct()
    if dct.is_random:
        dct.n = 2000
        nmin, nmax, nstep, nvalue = 1000, 7000, 1000, 1500
    else: 
        nmin, nmax, nstep, nvalue = 20, 100, 10, 30

    dct.n = nvalue

    @interact(nb_epochs=(200, 2000, 200), lr=[.001, 0.01, .1], nu=[0.001, 0.01, 0.1], 
       n=(nmin, nmax, nstep), fontsz=(12, 48, 4), rerun=true)
    def execute_run(nb_epochs=100, t1=1.0, nb_hid_layers=2, pts_per_layer=15, 
        n=nvalue, n_ic=30, n_bc=30, xl=3., nu=.01, lr=.1, is_random=true, rerun=true, fontsz=24):
        
        dct = globdct()
        print("n= ", n)
        print("is_random: ", dct.is_random)
        dct = globdct()  # might or might not have content
        dct.nu = nu
        dct.x0 = tt(0.)
        dct.xl = tt(xl)    # right spatial boundary
        dct.u0 = tt(0.)     # initial condition
        dct.ul = tt(0.)     # initial condition
        dct.t0 = t0 = tt(0.)
        dct.t1 = tt(t1)
        dct.n = n
        dct.n_ic = n_ic
        dct.n_bc = n_bc
        dct.lr = 0.001
        dct.fontsz = fontsz
        
        args = {"num_hidden": nb_hid_layers, "dim_hidden": pts_per_layer}
        # pinn initialized each time.
        nn_approximator = pinn(**args)
    
        # check that autograd derivatives match finite-difference derivatives
        # assert check_gradient(nn_approximator, x, t)
    
        # train the pinn
        loss_fn = partial(compute_loss)
        nn_trained = train_model(
            nn_approximator, loss_fn=loss_fn, 
            learning_rate=lr, 
            max_epochs=nb_epochs, 
        )
    
        plt.figure(1)
        plot_solution(nn_trained)
#----------------------------------------------------------------------
def plot_sol_losses(nb_epochs=100,  nb_hid_layers=3, pts_per_layer=30, fontsz=24,
    N=500, N_IC=30, N_BC=30, T1=1.0, xL=2., nu=.01, lr=.1, is_random=True, rerun=True, **kwargs):
    # print(f"nb_epochs={nb_epochs}")

    dct = GlobDct()  # might or might not have content
    dct.nu = nu
    dct.x0 = tt(0.)
    dct.xL = tt(xL)    # Right spatial boundary
    dct.u0 = tt(0.)     # Initial condition
    dct.uL = tt(0.)     # Initial condition
    dct.T0 = t0 = tt(0.)
    dct.T1 = tt(T1)
    dct.N = N
    dct.N_IC = N_IC
    dct.N_BC = N_BC
    dct.lr = 0.001
    dct.fontsz = fontsz
    dct.is_random = is_random
    
    args = {"num_hidden": nb_hid_layers, "dim_hidden": pts_per_layer}
    nn_approximator = PINN(**args)
    # Check that autograd derivatives match finite-difference derivatives
    # assert check_gradient(nn_approximator, x, t)

    # train the PINN
    loss_fn = partial(compute_loss)
    nn_trained = train_model(
        nn_approximator, loss_fn=loss_fn, 
        learning_rate=lr, 
        max_epochs=nb_epochs, 
    )

    plt.figure(1)
    plot_solution(nn_trained)
#---------------------------------------------------------------------------
def print_losses_impl(step=2):
        dct = GlobDct()
        sz = len(dct.total_losses)
        for i in range(0, sz, step):
            print(f"(epoch {i}) Interior, IC, BC loss: ",
                dct.total_losses[i], dct.IC_losses[i], dct.BC_losses[i])
#----------------------------------------------------------------------
def plot_run_tabs():
    dct = DictX()
    # dct.x0 = 0.
    # dct.xL = 2.
    height = {'orientation':'vertical', "layout":widgets.Layout(height='175px', margin='10px')}
    width = {"layout":widgets.Layout(width='175px', margin='20px')}

    nb_epochs = widgets.IntSlider(min=5, max=2005, step=200, value=50, description="#epochs", **height)
    nb_hidden_layers = widgets.IntSlider(min=2. , max=5., description='#layers', **height)
    pts_per_layer = widgets.IntSlider(min=10 , max=60, step=10, description='#pts in layer', **height)
    is_random = widgets.Checkbox(value=True, description='is_random', **width)

    if "is_random" in dct and dct.is_random:
        dct.N = 2000 # random (x,y) grid
    else: 
        dct.N = 30 # regular grid
        dct.is_random = False

    N = widgets.IntSlider(min=200 , max=10000, step=500, value=dct.N, description='N', **height)
    N_IC = widgets.IntSlider(min=20 , max=150, step=20, value=100, description='N_IC', **height)
    N_BC = widgets.IntSlider(min=20 , max=150, step=20, value=100, description='N_BC', **height)
    fontsz = widgets.IntSlider(min=12 , max=48, step=4, description='Font Size', **height)
    T1 = widgets.FloatSlider(min=0.5 , max=3., step=0.5, value=1., description='max(t)', **height)

    nu = widgets.Dropdown(options=[.001, .01, .1], description='nu', **width)
    lr = widgets.Dropdown(options=[.0001, .01, .1], description='lr', **width)

    ui = widgets.HBox([nb_epochs, nb_hidden_layers, pts_per_layer, 
                    N, N_IC, N_BC, fontsz, T1])
    dropboxes = widgets.HBox([nu, lr, is_random])

    # The plotting is in impl.execute_run

    # best approach to control the widget placement and styling
    # using interact() or interactive() duplicates some sliders
    out = widgets.interactive_output(plot_sol_losses, {'nb_epochs':nb_epochs,  
                   'nb_hid_layers':nb_hidden_layers, 'pts_per_layer':pts_per_layer, 'nu':nu, 
                    'lr':lr, 'N':N, 'r_IC':N_IC, 'N_BC':N_BC, 'fontsz':fontsz, 
                    'T1':T1, 'is_random':is_random})

    outW = widgets.VBox([ui, dropboxes, out])
    # return the widget for higher-level composition
    return outW
#---------------------------------------------------------------------------
def print_losses():
    step = widgets.IntSlider(min=1, max=10, description='Step')
    out = widgets.interactive_output(print_losses_impl, {'step':step})
    return widgets.VBox([step, out])
#-----------------------------------------------------------------------------
def plot_tab_structure():
    tab = widgets.Tab()
    out1 = plot_run_tabs()
    out2 = print_losses()
    children = [out1, out2]
    tab.children = children
    tab.set_title(0, "Plot")
    tab.set_title(1, "Text")
    return tab
#---------------------------------------------------------------------------
def unwrap(u: tt) -> np.ndarray:
    """ convert a 2D torch tensor to a 1D numpy array """
    return u.detach().numpy().reshape(-1)
#---------------------------------------------------------------------------
