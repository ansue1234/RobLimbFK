from robo_limb_ml.models.fk_rnn import FK_RNN
from robo_limb_ml.utils.data_loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import wandb
import time

# Argument parser setup
parser = argparse.ArgumentParser(description="Training FK_RNN model")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to load')
parser.add_argument('--train_data_path', type=str, default='../ml_data/train_data.csv', help='Path to the training data file')
parser.add_argument('--test_data_path', type=str, default='../ml_data/test_data.csv', help='Path to the testing data file')
parser.add_argument('--exp_name', type=str, default='', help='Things to add to experiment name')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--predict_len', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--prob_layer', type=bool, default=False)
parser.add_argument('--state', type=str, default='stateful')
parser.add_argument('--tag', type=str, default='debugging')
parser.add_argument('--vel', type=bool, default=False)
parser.add_argument('--no_time', type=bool, default=False)
args = parser.parse_args()


def get_loss(data_loader, model, hn, prev_setnum, loss_fn, optimizer, mode="train", state='stateful'):
    inputs, targets, set_num = data_loader.get_batch()
    if mode == 'train':
        optimizer.zero_grad()
        # print(inputs.shape)
    outputs, out_hn = model(inputs, hn.detach(), prob=args.prob_layer)
    loss = loss_fn(outputs, targets.detach())
    if mode == 'train':
        loss.backward()
        optimizer.step()
    loss_batch = loss.item()
    if set_num != prev_setnum or state != 'stateful':
        hn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
    else:
        hn = out_hn
    return hn, loss_batch, set_num


if __name__ == "__main__":
    # print("Hello")
    experiment_name = "RNN_b{}_e{}_s{}_len{}_{}_{}".format(args.batch_size, args.epochs, args.num_samples, args.seq_len, args.exp_name, int(time.time()))
    wandb.init(
        # set the wandb project where this run will be logged
        project="RobLimbFK",
        entity="gsue",
        # track hyperparameters and run metadata
        config = {"architecture": "RNN",
                  "num_samples": args.num_samples,
                  "batch_size": args.batch_size,
                  "epochs": args.epochs,
                  "num_layers": args.num_layers,
                  "hidden_size": args.hidden_size,
                  "seq_len": args.seq_len,
                  "prob": args.prob_layer,
                  "seed": args.seed
        },
        name=experiment_name,
        tags=[args.tag]
    )
    # print("Hi")
    # train_data_path = '../ml_data/train_data.csv'
    # test_data_path = '../ml_data/test_data.csv'
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(args.prob_layer)
    print(args.state)
    input_features = ['time_begin',
                      'time_begin_traj',
                      'theta_x',
                      'theta_y',
                      'vel_x',
                      'vel_y',
                      'X_throttle',
                      'Y_throttle'] 
    if args.no_time:
        input_features.remove('time_begin')
        input_features.remove('time_begin_traj')
    if not args.vel:
        input_features.remove('vel_x')
        input_features.remove('vel_y')
        
    train_data_loader = DataLoader(file_path=args.train_data_path,
                                   batch_size=args.batch_size,
                                   device=device,
                                   predict_len=args.predict_len,
                                   seq_len=args.seq_len,
                                   num_samples=args.num_samples,
                                   input_features=input_features,
                                   pad=True)
    test_data_loader = DataLoader(file_path=args.test_data_path,
                                  batch_size=args.batch_size,
                                  device=device,
                                  predict_len=args.predict_len,
                                  num_samples=-1,
                                  input_features=input_features,
                                  seq_len=args.seq_len,
                                  pad=True)
    input_size = train_data_loader.input_dim
    hidden_size = args.hidden_size
    output_size = train_data_loader.output_dim
    num_layers = args.num_layers
    
    # print(train_data_loader.n_samples)
    # print(train_data_loader.batch_size)

    model = FK_RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_size=args.batch_size,
                    output_size=output_size,
                    device=device,
                    batch_first=True).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    
    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0
        set_num = 0
        model.train()
        cn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        hn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        for batch in range(train_data_loader.n_batches):
            hn, loss, set_num = get_loss(train_data_loader, model, hn, set_num, loss_fn, optimizer, mode="train", state=args.state)
            loss_epoch += loss
        wandb.log({"loss_epoch": loss_epoch, "epoch": epoch})
        wandb.log({"Loss_epoch_per_batch": loss_epoch/train_data_loader.get_n_batches(), "epoch": epoch})
        # evauation
        loss_evals = 0
        eval_set_n = 0
        model.eval()
        cn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        hn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        with torch.no_grad():
            for batch in range(test_data_loader.n_batches):
                hn, loss, set_num = get_loss(train_data_loader, model, hn, set_num, loss_fn, optimizer, mode="test", state=args.state)
                loss_evals += loss
            wandb.log({"val_loss": loss_evals, "epoch": epoch})
            wandb.log({"val_loss_per_batch": loss_evals/test_data_loader.get_n_batches(), "epoch": epoch})

    torch.save(model.state_dict(), '../model_weights/new_weights/'+experiment_name)
    



            



