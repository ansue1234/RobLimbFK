from robo_limb_ml.models.fk_lstm import FK_LSTM
from robo_limb_ml.utils.data_loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import wandb

torch.manual_seed(1)
# Argument parser setup
parser = argparse.ArgumentParser(description="Training FK_LSTM model")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
parser.add_argument('--num_samples', type=int, default=2**14, help='Number of samples to load')
parser.add_argument('--data_path', type=str, default='../data/data.csv', help='Path to the training data file')
parser.add_argument('--exp_name', type=str, default='', help='Things to add to experiment name')
args = parser.parse_args()

if __name__ == "__main__":
    # print("Hello")
    wandb.init(
        # set the wandb project where this run will be logged
        project="RobLimbFK",
        entity="gsue",
        # track hyperparameters and run metadata
        config = {"architecture": "LSTM",
                  "num_samples": args.num_samples,
                  "batch_size": args.batch_size,
                  "epochs": args.epochs,
        },
        name="LSTM_b{}_e{}_s{}_{}".format(args.batch_size, args.epochs, args.num_samples, args.exp_name)
    )
    # print("Hi")

    input_size = 11
    hidden_size = 32
    num_layers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FK_LSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    device=device,
                    batch_first=True)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    batch_size = args.batch_size
    num_samples = args.num_samples
    # print("hi")
    # Load data
    data_loader = DataLoader(file_path=args.data_path,
                             batch_size=batch_size,
                             num_samples=num_samples,
                             device=device)
    iterations = 0
    testing_data_loader = DataLoader(file_path='../data/test_data.csv',
                                     num_samples=args.num_samples//4,
                                     batch_size=batch_size,
                                     device=device)
    
    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0
        model.train()
        for batch in range(data_loader.get_n_batches()):
            inputs, targets = data_loader.get_batch()
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs, prob=True)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            iterations += batch_size
            # print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}')
            # wandb.log({"loss": loss.item()})
        wandb.log({"loss_epoch": loss_epoch, "epoch": epoch})
        wandb.log({"Loss_epoch_per_batch": loss_epoch/data_loader.get_n_batches(), "epoch": epoch})
        model.eval()
        with torch.no_grad():
            inputs_eval, targets_eval = testing_data_loader.get_all_data()
            outputs_eval = model(inputs_eval, prob=True)
            loss_eval = loss_fn(outputs_eval, targets_eval)
            wandb.log({"val_loss": loss_eval.item(), "epoch": epoch})

    torch.save(model.state_dict(), '../model_weights/'+args.exp_name)
    



            



