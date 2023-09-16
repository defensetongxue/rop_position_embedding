import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, meta in train_loader:
        # Moving inputs and targets to the correct device
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)

def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer
def get_lr_scheduler(optimizer, cfg):
    if cfg['method'] == 'reduce_plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg['reduce_plateau_patience'],
            factor=cfg['reduce_plateau_factor'],
            cooldown=cfg['cooldown'],
            verbose=False
        )
    elif cfg['method'] == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['cosine_annealing_T_max'])
    elif cfg['method'] == 'constant':
        lr_scheduler = None  # No learning rate scheduling for constant LR
    else:
        raise ValueError("Invalid learning rate scheduling method")
    
    return lr_scheduler
