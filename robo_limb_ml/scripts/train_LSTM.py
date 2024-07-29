from robo_limb_ml.models.fk_lstm import FK_LSTM
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
parser = argparse.ArgumentParser(description="Training FK_LSTM model")
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
args = parser.parse_args()


def get_loss(data_loader, model, hn, cn, prev_setnum, loss_fn, optimizer, mode="train", state='stateful'):
    inputs, targets, set_num = data_loader.get_batch()
    if mode == 'train':
        optimizer.zero_grad()
        # print(inputs.shape)
    outputs, out_cn, out_hn = model(inputs, hn.detach(), cn.detach(), prob=args.prob_layer)
    loss = loss_fn(outputs, targets.detach())
    if mode == 'train':
        loss.backward()
        optimizer.step()
    loss_batch = loss.item()
    if set_num != prev_setnum or state != 'stateful':
        cn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        hn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
    else:
        cn = out_cn
        hn = out_hn
    return hn, cn, loss_batch, set_num


if __name__ == "__main__":
    # print("Hello")
    experiment_name = "LSTM_b{}_e{}_s{}_{}_{}".format(args.batch_size, args.epochs, args.num_samples, args.exp_name, int(time.time()))
    wandb.init(
        # set the wandb project where this run will be logged
        project="RobLimbFK",
        entity="gsue",
        # track hyperparameters and run metadata
        config = {"architecture": "LSTM",
                  "num_samples": args.num_samples,
                  "batch_size": args.batch_size,
                  "epochs": args.epochs,
                  "num_layers": args.num_layers,
                  "hidden_size": args.hidden_size,
                  "seq_len": args.seq_len,
                  "prob": args.prob_layer,
                  "seed": args.seed
        },
        name=experiment_name
    )
    # print("Hi")
    # train_data_path = '../ml_data/train_data.csv'
    # test_data_path = '../ml_data/test_data.csv'
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(args.prob_layer)
    train_data_loader = DataLoader(file_path=args.train_data_path,
                                   batch_size=args.batch_size,
                                   device=device,
                                   predict_len=args.predict_len,
                                   seq_len=args.seq_len,
                                   num_samples=args.num_samples,
                                   pad=True)
    test_data_loader = DataLoader(file_path=args.test_data_path,
                                  batch_size=args.batch_size,
                                  device=device,
                                  predict_len=args.predict_len,
                                  num_samples=-1,
                                  seq_len=args.seq_len,
                                  pad=True)
    input_size = train_data_loader.input_dim
    hidden_size = args.hidden_size
    output_size = train_data_loader.output_dim
    num_layers = args.num_layers
    
    # print(train_data_loader.n_samples)
    # print(train_data_loader.batch_size)

    model = FK_LSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_size=args.batch_size,
                    output_size=output_size,
                    device=device,
                    batch_first=True).to(device=device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    # batch_size = args.batch_size
    # num_samples = args.num_samples
    # # print("hi")
    # # Load data
    # data_loader = DataLoader(file_path=args.data_path,
    #                          batch_size=batch_size,
    #                          num_samples=num_samples,
    #                          device=device)
    # testing_data_loader = DataLoader(file_path='../data/test_data.csv',
    #                                  num_samples=args.num_samples//4,
    #                                  batch_size=batch_size,
    #                                  device=device)
    
    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0
        set_num = 0
        model.train()
        cn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        hn = torch.zeros(num_layers, args.batch_size, hidden_size).to(device)
        for batch in range(train_data_loader.n_batches):
            hn, cn, loss, set_num = get_loss(train_data_loader, model, hn, cn, set_num, loss_fn, optimizer, mode="train", state='stateful')
            loss_epoch += loss
            # print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}')
            # wandb.log({"loss": loss.item()})
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
                hn, cn, loss, set_num = get_loss(train_data_loader, model, hn, cn, set_num, loss_fn, optimizer, mode="test", state='stateful')
                loss_evals += loss
            wandb.log({"val_loss": loss_evals, "epoch": epoch})
            wandb.log({"val_loss_per_batch": loss_evals/test_data_loader.get_n_batches(), "epoch": epoch})

    torch.save(model.state_dict(), '../model_weights/'+experiment_name)
    



            



