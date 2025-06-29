import torch
from itertools import cycle
from signal_sep_pinns.utils import *


###############################################################################################################
# Function to train PINNslope with derivative term loss 
# ###############################################################################################################



def train_grr_pinnslope(models, 
                        optimizers, 
                        data_term_criterion, 
                        lamdas,  
                        grid_data_loader, 
                        traces_data_loader, 
                        derivative_scaling, 
                        c, 
                        first_derivative=False, 
                        second_derivative=False 
                        ):

    """Training step

    Performs a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    data_term_criterion : :obj:`torch.nn.modules.loss`
        Loss data term function
    optimizer : :obj:`torch.optim`
        Optimizer
    lamdas : :obj:`list`
        loss function for data term weights 'int': loss[0] = pressure data term, loss[1] = acceleration data term
    grid_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for computational grid
    traces_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for available traces
    c : :obj:`float`
        Value to constrain the sigmoid function output and the overall slope prediction
    
    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset

    """

    # Retrieve models:
    model, model_slope = models[0], models[1]

    # Retrieve optims:
    optimizer, optimizer_slope = optimizers[0], optimizers[1]

    # Retrieve loss function weights:
    lamda, lamda2 = lamdas[0], lamdas[1]

    # Retrieve traces batch size (needed later for array slicing):  
    batch_traces_size = traces_data_loader.batch_size
    
    # Constrain output:
    sigmoid = torch.nn.Sigmoid()
    c = torch.tensor( c ) # reflections prediction

    # Set losses to 0 for the start:
    training_loss = 0
    loss_data_values = 0
    loss_derivatives_values = 0
    loss_phy_values = 0

    model.train()
    model_slope.train()

    for a, b in zip( grid_data_loader, cycle( traces_data_loader ) ):
        
        optimizer.zero_grad()
        
        input_grid_points = a[0][:,:2]        # grid-points: (t_i, x_i)
        traces_grid_points = b[0][:,:2]       # grid-points: (t_j, x_j)
        gt_traces = (b[0][:,2]).view(-1,1)    # available data: u(t_j, x_j)
        
        input = (torch.cat((input_grid_points, traces_grid_points), axis=0)).requires_grad_(True)
        
        # Wavefield network prediction:
        u = model(input)

        # Compute derivatives for plane-wave PDE:
        gradient = torch.autograd.grad(u, 
                                    input, 
                                    torch.ones_like(u), 
                                    create_graph=True
                                    )[0]
        dx = gradient[: (input.shape[0] - (batch_traces_size)), 0]
        dt = gradient[: (input.shape[0] - (batch_traces_size)), 1]
        
        # Derivative of the predicted data:
        if first_derivative == True :
            # time derivative data: u'(t_j, x_j)
            dudt_traces = (b[0][:,3]).view(-1,1)  
            data_dt1 = gradient[(input.shape[0] - (batch_traces_size)): , 1] * derivative_scaling
            
            # L1 derivative data fitting, || phi'_2(tj, xj) - u'(tj, xj) || :
            data_dt1 = ( data_dt1 ).view(-1,1)
            loss_derivative_data = data_term_criterion( data_dt1 , dudt_traces )
            loss_derivatives_values += loss_derivative_data.item()

        elif second_derivative == True :
            # time derivative data: u''(t_j, x_j)
            du2dt2_traces = (b[0][:,3]).view(-1,1)
            dgradient1_dt =  torch.autograd.grad(gradient[:,1].view(-1,1) * derivative_scaling, 
                                                input, 
                                                torch.ones_like(u), 
                                                create_graph=True
                                                )[0]
            data_dt2 = dgradient1_dt[(input.shape[0] - (batch_traces_size)): , 1] * derivative_scaling
            
            # L1 derivative data fitting, || phi'_2(tj, xj) - u''(tj, xj) || :
            data_dt2 = ( data_dt2 ).view(-1,1)
            loss_derivative_data = data_term_criterion( data_dt2 , du2dt2_traces )
            loss_derivatives_values += loss_derivative_data.item()


        # Slope network prediction:
        model_slope.train()
        optimizer_slope.zero_grad()
        sigma = sigmoid( model_slope(input_grid_points) ) * c
        
        # Plane-wave PDE
        loss_phy = torch.mean((dx + sigma[:,0]*dt)**2)
        loss_phy_values += loss_phy.item()
        
        # L1 Data fitting term:
        loss_data = data_term_criterion(u[(input.shape[0] - (batch_traces_size)):], gt_traces)
        loss_data_values += loss_data.item()
        
        # Total Loss function:
        if first_derivative or second_derivative == True:
            loss = lamda*loss_data + loss_phy + lamda2*loss_derivative_data
            training_loss += loss.item()   
        else:
            loss = lamda*loss_data + loss_phy   
            training_loss += loss.item() 

        # Calculate the gradient:
        loss.backward(retain_graph=True)

    training_loss /= len(grid_data_loader)
    loss_data_values /= len(grid_data_loader)
    loss_derivatives_values /= len(grid_data_loader)
    loss_phy_values /= len(grid_data_loader)

    return training_loss, loss_data_values, loss_derivatives_values, loss_phy_values







###############################################################################################################
# Function to train PINNslope for double slope estimation 
# ###############################################################################################################



def train_double_pinnslope(models, 
                        optimizers, 
                        data_term_criterion, 
                        lamdas,  
                        grid_data_loader, 
                        traces_data_loader, 
                        c2, 
                        ):

    """Training step

    Performs a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    data_term_criterion : :obj:`torch.nn.modules.loss`
        Loss data term function
    optimizer : :obj:`torch.optim`
        Optimizer
    lamdas : :obj:`list`
        loss function for data term weights 'int': loss[0] = pressure data term, loss[1] = acceleration data term
    grid_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for computational grid
    traces_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for available traces
    c : :obj:`float`
        Value to constrain the sigmoid function output and the overall slope prediction
    
    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset

    """

    # Retrieve models:
    model, model_slope = models[0], models[1]

    # Retrieve optims:
    optimizer, optimizer_slope = optimizers[0], optimizers[1]

    # Retrieve loss function weights:
    lamda = lamdas[0]

    # Retrieve traces batch size (needed later for array slicing):  
    batch_traces_size = traces_data_loader.batch_size
    
    # Constrain output:
    sigmoid = torch.nn.Sigmoid()
    c2 = torch.tensor( c2 ) # reflections prediction

    # Set losses to 0 for the start:
    training_loss = 0
    loss_data_values = 0
    loss_phy_values = 0

    model.train()
    model_slope.train()

    for a, b in zip( grid_data_loader, cycle( traces_data_loader ) ):
        
        optimizer.zero_grad()
        
        input_grid_points = a[0][:,:2];      # grid-points: (t_i, x_i)
        traces_grid_points = b[0][:,:2];     # grid-points: (t_j, x_j)
        gt_traces = (b[0][:,2]).view(-1,1);  # available data: u(t_j, x_j)
        
        input = (torch.cat((input_grid_points, traces_grid_points), axis=0)).requires_grad_(True)

        # Wavefield network prediction:
        u = model(input)

        # Compute derivatives for plane-wave PDE:
        # Output 1
        gradient1 = torch.autograd.grad(u[:,0], 
                                        input, 
                                        torch.ones_like(u[:,0]), 
                                        create_graph=True
                                        )[0]
        dx1 = gradient1[: (input.shape[0] - (batch_traces_size)), 0]
        dt1 = gradient1[: (input.shape[0] - (batch_traces_size)), 1]
        # Output 2
        gradient2 = torch.autograd.grad(u[:,1], 
                                        input, 
                                        torch.ones_like(u[:,1]), 
                                        create_graph=True
                                        )[0]
        dx2 = gradient2[: (input.shape[0] - (batch_traces_size)), 0]
        dt2 = gradient2[: (input.shape[0] - (batch_traces_size)), 1]

        # Slope network prediction:
        model_slope.train()
        optimizer_slope.zero_grad()
        sigma = model_slope(input_grid_points)
        
        # Apply constraint c2 to slope network output:
        sigma1 = sigmoid(sigma[:,0])
        sigma2 = sigmoid(sigma[:,1]) * c2
            
        # plane-wave PDE(1) + plane-wave PDE(2):
        loss_phy = torch.mean((dx1 + sigma1*dt1)**2) +  torch.mean((dx2 + sigma2*dt2)**2)
        loss_phy_values += loss_phy.item()
        
        # L1 Data fitting term || (phi1(t_j,x_j) + phi2(t_j,x_j)) - u(t_j,x_j) || :
        loss_data = data_term_criterion( ( u[(input.shape[0] - (batch_traces_size)):, :1].squeeze() + u[(input.shape[0] - (batch_traces_size)):, 1].squeeze() ).view(-1,1),  gt_traces )
        loss_data_values += loss_data.item()
        
        # Total Loss function:
        loss = lamda*loss_data + loss_phy
        training_loss += loss.item()         
        
        # Calculate the gradient:
        loss.backward(retain_graph=True)


    training_loss /= len(grid_data_loader)
    loss_data_values /= len(grid_data_loader)
    loss_phy_values /= len(grid_data_loader)

    return training_loss, loss_data_values, loss_phy_values




###############################################################################################################
# Function to train PINNslope (Brandolin et al. 2024) 
# ###############################################################################################################

def train_pinnslope(models, optimizers, data_term_criterion, lamdas,  grid_data_loader, traces_data_loader):
    """Training step

    Performs a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    data_term_criterion : :obj:`torch.nn.modules.loss`
        Loss data term function
    optimizer : :obj:`torch.optim`
        Optimizer
    lamdas : :obj:`list`
        loss function for data term weights 'int': loss[0] = pressure data term, loss[1] = acceleration data term
    grid_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for computational grid
    traces_data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training DataLoader for available traces

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset

    """
    
    # Retrieve models:
    model, model_slope = models[0], models[1]

    # Retrieve optims:
    optimizer, optimizer_slope = optimizers[0], optimizers[1]

    # Retrieve loss function weights:
    lamda = lamdas[0]

    # Retrieve traces batch size (needed later for array slicing):  
    batch_traces_size = traces_data_loader.batch_size
    
    # Set losses to 0 for the start:
    training_loss = 0
    loss_data_values = 0
    loss_phy_values = 0

    model.train()
    model_slope.train()

    for a, b in zip( grid_data_loader, cycle( traces_data_loader ) ):
        
        optimizer.zero_grad()

        input_grid_points = a[0][:,:2];        # grid-points: (t_i, x_i)
        traces_grid_points = b[0][:,:2];       # grid-points: (t_j, x_j)
        u_gt_traces = (b[0][:,2]).view(-1,1);  # pressure traces: u(t_j, x_j)
        
        input = ( torch.cat((input_grid_points, traces_grid_points), axis=0) ).requires_grad_(True)

        # Wavefield network prediction:
        phi = model(input)

        #Compute derivatives for plane-wave PDE:
        phi_gradient = torch.autograd.grad(phi, 
                                        input, 
                                        torch.ones_like(phi), 
                                        create_graph=True
                                        )[0]
        dx = phi_gradient[: (input.shape[0] - (batch_traces_size)), 0]
        dt = phi_gradient[: (input.shape[0] - (batch_traces_size)), 1]

        # Slope network prediction:
        optimizer_slope.zero_grad()
        sigma = model_slope(input_grid_points)

        # Plane-wave PDE
        loss_phy = torch.mean((dx + sigma[:,0]*dt)**2)
        loss_phy_values += loss_phy.item()
        
        # L1 Data fitting term || phi_u(t_j,x_j) - u(t_j,x_j) || :
        phi_u_traces = phi[(input.shape[0] - (batch_traces_size)):]
        loss_data = data_term_criterion( phi_u_traces, u_gt_traces )
        loss_data_values += loss_data.item()

        # Total Loss function:
        loss = loss_phy + lamda*loss_data
        training_loss += loss.item() 
        
        # Calculate the gradient:
        loss.backward(retain_graph=True)
    
        optimizer.step()
        optimizer_slope.step()

    training_loss /= len(grid_data_loader)
    loss_data_values /= len(grid_data_loader)
    loss_phy_values /= len(grid_data_loader)

    return training_loss, loss_data_values, loss_phy_values