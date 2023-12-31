import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_instance, train_epoch, val_epoch,get_optimizer,losses,lr_sche
from utils_ import rop_posembed_dataset as CustomDatset
import models
import os
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
criterion=get_instance(losses,args.configs['model']['loss_func'])
if os.path.isfile(args.from_checkpoint):
    print(f"loadding the exit checkpoints {args.from_checkpoint}")
    model.load_state_dict(
    torch.load(args.from_checkpoint))
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDatset(args.data_path,'train',split_name=args.split_name,image_resize=args.configs['image_resize'])
val_dataset=CustomDatset(args.data_path,'val',split_name=args.split_name,image_resize=args.configs['image_resize'])
# Create the data loaders
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'])
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("There is {} patch size".format(args.configs['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs['train']['end_epoch']
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss = val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{total_epoches}," 
          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}," 
            f" Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}" )
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}"))
        print("Model saved as {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}")))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break
