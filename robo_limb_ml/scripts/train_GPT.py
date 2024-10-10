import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import wandb
import time
import math

# Import the transformer model
from robo_limb_ml.models.fk_attention import FK_GPT
from robo_limb_ml.utils.data_loader import DataLoader

# Argument parser setup
parser = argparse.ArgumentParser(description="Training Transformer model")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to load')
parser.add_argument('--train_data_path', type=str, default='../ml_data/train_data.csv', help='Path to the training data file')
parser.add_argument('--test_data_path', type=str, default='../ml_data/test_data.csv', help='Path to the testing data file')
parser.add_argument('--exp_name', type=str, default='', help='Things to add to experiment name')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--predict_len', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--embed_dim', type=int, default=128)  # Embedding dimension for the transformer
parser.add_argument('--num_heads', type=int, default=1)    # Number of attention heads
parser.add_argument('--num_layers', type=int, default=1)   # Number of transformer layers
parser.add_argument('--tag', type=str, default='debugging')
parser.add_argument('--rollout', type=bool, default=True)
parser.add_argument('--sin_pos', type=bool, default=True)
args = parser.parse_args()

def prep_inputs(prev_inputs, outputs, throttles):
    next_step = torch.cat((outputs, throttles), dim=2)
    inputs = torch.cat((prev_inputs, next_step), dim=1)
    return inputs

def get_loss(data_loader, model, loss_fn, optimizer, mode="train", rollout=True):
    if rollout:
        inputs, targets, throttle, set_num = data_loader.get_batch_rollout()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets[:, 0, :].unsqueeze(1).detach())
        rollout_input = prep_inputs(inputs[:, 1:, :], outputs, throttle[:, 0, :].unsqueeze(1))
        pred_outputs = outputs
        for i in range(2, targets.shape[1]):
            outputs = model(rollout_input)
            pred_outputs = torch.cat((pred_outputs, outputs), dim=1)
            rollout_input = prep_inputs(rollout_input[:, 1:, :], outputs[:, -1, :].unsqueeze(1), throttle[:, i, :].unsqueeze(1))
        loss = loss_fn(pred_outputs, targets.detach())
    else:
        inputs, targets, set_num = data_loader.get_batch()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets.detach())
        
    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
    loss_batch = loss.item()
    return loss_batch, set_num

if __name__ == "__main__":
    experiment_name = "Transformer_b{}_e{}_s{}_len{}_{}_{}".format(
        args.batch_size, args.epochs, args.num_samples, args.seq_len, args.exp_name, int(time.time()))
    wandb.init(
        project="RobLimbFK",
        entity="gsue",
        config={
            "architecture": "Transformer",
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_layers": args.num_layers,
            "embed_dim": args.embed_dim,
            "seq_len": args.seq_len,
            "pred_len":args.predict_len,
            "seed": args.seed,
        },
        name=experiment_name,
        tags=[args.tag]
    )

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    input_features = [
        'theta_x',
        'theta_y',
        'vel_x',
        'vel_y',
        'X_throttle',
        'Y_throttle'
    ]

    # DataLoader should return inputs and targets suitable for the transformer
    train_data_loader = DataLoader(
        file_path=args.train_data_path,
        batch_size=args.batch_size,
        device=device,
        predict_len=args.predict_len,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        input_features=input_features,
        pad=True
    )
    test_data_loader = DataLoader(
        file_path=args.test_data_path,
        batch_size=args.batch_size,
        device=device,
        predict_len=args.predict_len,
        num_samples=-1,
        input_features=input_features,
        seq_len=args.seq_len,
        pad=True
    )
    input_size = train_data_loader.input_dim
    output_size = train_data_loader.output_dim

    # Initialize the transformer model
    model = FK_GPT(
        input_dim=input_size,
        embed_dim=args.embed_dim,
        out_dim=output_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.seq_len,
        device=device,
        sin_pos=args.sin_pos
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(args.epochs)):
        loss_epoch = 0
        set_num = 0
        model.train()
        for batch in range(train_data_loader.n_batches):
            # Adjusted to remove LSTM-specific logic
            loss, set_num = get_loss(train_data_loader, model, loss_fn, optimizer, mode="train", rollout=args.rollout)
            loss_epoch += loss
        wandb.log({"loss_epoch": loss_epoch, "epoch": epoch})
        wandb.log({"Loss_epoch_per_batch": loss_epoch / train_data_loader.get_n_batches(), "epoch": epoch})

        # Evaluation
        loss_evals = 0
        model.eval()
        with torch.no_grad():
            for batch in range(test_data_loader.n_batches):
                loss, set_num = get_loss(test_data_loader, model, loss_fn, optimizer, mode="test", rollout=args.rollout)
                loss_evals += loss
            wandb.log({"val_loss": loss_evals, "epoch": epoch})
            wandb.log({"val_loss_per_batch": loss_evals / test_data_loader.get_n_batches(), "epoch": epoch})

    torch.save(model.state_dict(), '../model_weights/new_weights/' + experiment_name)
