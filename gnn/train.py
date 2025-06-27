```python
import os
import time
import torch
import torch.nn as nn
from alg.opt import get_optimizer_adamw
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_args, print_environ, train_valid_target_eval_names, alg_loss_dict
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from gnn.temporal_gcn import EnhancedTemporalGCN
from gnn.graph_builder import GraphBuilder

# ======================= DOMAIN ADVERSARIAL LOSS =======================
class DomainAdversarialLoss(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, features, labels):
        preds = self.classifier(features).squeeze()
        return self.loss_fn(preds, labels.float())

# ======================= MAIN TRAINING FUNCTION =======================
def main(args):
    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    os.makedirs(args.output, exist_ok=True)

    # Load data (returns PyTorch Geometric or Torch DataLoaders)
    train_loader, train_loader_ns, val_loader, test_loader, tr, val, test = get_act_dataloader(args)[:7]

    # Initialize Diversify algorithm
    AlgoClass = alg.get_algorithm_class(args.algorithm)
    algorithm = AlgoClass(args).to(args.device)

    # ======================= GNN INTEGRATION =======================
    if args.use_gnn:
        print("Initializing GNN feature extractor...")
        builder = GraphBuilder(
            method='correlation', threshold_type='adaptive', default_threshold=0.3, adaptive_factor=1.5
        )
        gnn = EnhancedTemporalGCN(
            input_dim=8,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=builder,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            bidirectional=args.bidirectional,
            lstm_dropout=args.lstm_dropout,
            n_layers=args.gnn_layers,
            use_tcn=args.use_tcn
        ).to(args.device)
        algorithm.featurizer = gnn

        # Bottleneck projections
        def make_bottleneck(in_dim, out_dim, layers):
            try:
                n = int(layers)
                mods = []
                for _ in range(n - 1): mods += [nn.Linear(in_dim, in_dim), nn.ReLU()]
                mods.append(nn.Linear(in_dim, out_dim))
                return nn.Sequential(*mods)
            except:
                return nn.Linear(in_dim, out_dim)

        in_dim, out_dim = args.gnn_output_dim, int(args.bottleneck)
        algorithm.bottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)
        algorithm.abottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)
        algorithm.dbottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)

    algorithm.train()

    # Optimizers and scheduler
    optimizer = get_optimizer_adamw(algorithm, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    # Domain adversarial loss
    if args.domain_adv_weight > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(int(args.bottleneck)).to(args.device)

    # Logging containers
    logs = {k: [] for k in ['train_acc', 'val_acc', 'test_acc', 'class_loss', 'dis_loss', 'ent_loss', 'total_loss']}
    best_val = 0.0

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        epoch_start = time.time()
        # 1) Feature update
        for batch in train_loader:
            inputs, labels, domains = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
            loss_dict = algorithm.update_a([inputs, labels, domains], optimizer)
            logs['class_loss'].append(loss_dict['class'])

        # 2) Latent domain characterization
        for batch in train_loader:
            inputs, labels, domains = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
            loss_d = algorithm.update_d([inputs, labels, domains], optimizer)
            logs['dis_loss'].append(loss_d['dis']); logs['ent_loss'].append(loss_d['ent']); logs['total_loss'].append(loss_d['total'])

        # 3) Domain-invariant feature learning
        for batch in train_loader:
            inputs, labels, domains = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
            _ = algorithm.update([inputs, labels, domains], optimizer)

        # Evaluate
        eval_fn = modelopera.accuracy
        logs['train_acc'].append(eval_fn(algorithm, train_loader_ns, None))
        logs['val_acc'].append(eval_fn(algorithm, val_loader, None))
        logs['test_acc'].append(eval_fn(algorithm, test_loader, None))

        # Scheduler step
        scheduler.step()

        # Save best
        if logs['val_acc'][-1] > best_val:
            best_val = logs['val_acc'][-1]
            torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))

        print(f"Epoch {epoch}/{args.max_epoch} — Train: {logs['train_acc'][-1]:.4f}, Val: {logs['val_acc'][-1]:.4f}, Test: {logs['test_acc'][-1]:.4f}, Time: {time.time()-epoch_start:.1f}s")

    print(f"Training complete. Best-val acc: {best_val:.4f}")

if __name__ == '__main__':
    args = get_args()
    # GNN defaults
    if not hasattr(args, 'use_gnn'): args.use_gnn = False
    if args.use_gnn:
        args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
        args.gnn_output_dim = getattr(args, 'gnn_output_dim', 256)
        args.gnn_layers = getattr(args, 'gnn_layers', 3)
        args.use_tcn = getattr(args, 'use_tcn', True)
        args.lstm_hidden_size = getattr(args, 'lstm_hidden_size', 128)
        args.lstm_layers = getattr(args, 'lstm_layers', 1)
        args.bidirectional = getattr(args, 'bidirectional', False)
        args.lstm_dropout = getattr(args, 'lstm_dropout', 0.2)
    main(args)
```
