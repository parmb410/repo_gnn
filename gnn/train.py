import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_args, print_environ
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from gnn.temporal_gcn import TemporalGCN
from gnn.graph_builder import GraphBuilder

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

def fix_emg_shape(x):
    """
    Ensures input x is [batch, 8, 200] for GNN.
    Accepts: [batch, 1, 200], [batch, 200, 1], [batch, 8, 200], [batch, 200, 8].
    Returns: [batch, 8, 200]
    """
    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            if x.shape[1] == 1 and x.shape[2] == 200:
                # [batch, 1, 200] -> [batch, 8, 200] (repeat across channels)
                x = x.repeat(1, 8, 1)
            elif x.shape[1] == 200 and x.shape[2] == 1:
                # [batch, 200, 1] -> [batch, 200, 8], then permute to [batch, 8, 200]
                x = x.repeat(1, 1, 8).permute(0, 2, 1)
            elif x.shape[1] == 8 and x.shape[2] == 200:
                pass  # correct shape
            elif x.shape[1] == 200 and x.shape[2] == 8:
                x = x.permute(0, 2, 1)
        # If PyG Batch object, skip
    return x

def main(args):
    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    os.makedirs(args.output, exist_ok=True)

    # Load data loaders
    train_loader, train_ns_loader, val_loader, test_loader, _, _, _ = get_act_dataloader(args)

    # Initialize Diversify algorithm
    AlgoClass = alg.get_algorithm_class(args.algorithm)
    algorithm = AlgoClass(args).to(args.device)

    # GNN integration
    if args.use_gnn:
        print("Initializing GNN feature extractor...")
        graph_builder = GraphBuilder(
            method='correlation', threshold_type='adaptive',
            default_threshold=0.3, adaptive_factor=1.5
        )
        gnn = TemporalGCN(
            input_dim=8,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            graph_builder=graph_builder
        ).to(args.device)
        algorithm.featurizer = gnn

        def make_bottleneck(in_dim, out_dim, layers):
            try:
                num = int(layers)
                modules = []
                for _ in range(num - 1):
                    modules += [nn.Linear(in_dim, in_dim), nn.ReLU()]
                modules.append(nn.Linear(in_dim, out_dim))
                return nn.Sequential(*modules)
            except Exception:
                return nn.Linear(in_dim, out_dim)

        in_dim, out_dim = args.gnn_output_dim, int(args.bottleneck)
        algorithm.bottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)
        algorithm.abottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)
        algorithm.dbottleneck = make_bottleneck(in_dim, out_dim, args.layer).to(args.device)

    algorithm.train()

    optimizer = optim.AdamW(
        algorithm.parameters(), lr=args.lr, weight_decay=getattr(args, 'weight_decay', 0)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    if getattr(args, 'domain_adv_weight', 0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(int(args.bottleneck)).to(args.device)

    logs = {k: [] for k in ['train_acc', 'val_acc', 'test_acc', 'class_loss', 'dis_loss', 'ent_loss', 'total_loss']}
    best_val = 0.0

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()
        # 1) Feature update
        for x, y, d in train_loader:
            if args.use_gnn:
                x = x.to(args.device)
                x = fix_emg_shape(x)
            else:
                x = x.to(args.device).float()
            y = y.to(args.device)
            d = d.to(args.device)
            res = algorithm.update_a([x, y, d], optimizer)
            logs['class_loss'].append(res['class'])
        # 2) Latent domain characterization
        for x, y, d in train_loader:
            if args.use_gnn:
                x = x.to(args.device)
                x = fix_emg_shape(x)
            else:
                x = x.to(args.device).float()
            y = y.to(args.device)
            d = d.to(args.device)
            res = algorithm.update_d([x, y, d], optimizer)
            logs['dis_loss'].append(res['dis'])
            logs['ent_loss'].append(res['ent'])
            logs['total_loss'].append(res['total'])
        # 3) Domain-invariant feature learning
        for x, y, d in train_loader:
            if args.use_gnn:
                x = x.to(args.device)
                x = fix_emg_shape(x)
            else:
                x = x.to(args.device).float()
            y = y.to(args.device)
            d = d.to(args.device)
            _ = algorithm.update([x, y, d], optimizer)

        # Evaluation
        acc_fn = modelopera.accuracy
        logs['train_acc'].append(acc_fn(algorithm, train_ns_loader, None))
        logs['val_acc'].append(acc_fn(algorithm, val_loader, None))
        logs['test_acc'].append(acc_fn(algorithm, test_loader, None))
        scheduler.step()

        if logs['val_acc'][-1] > best_val:
            best_val = logs['val_acc'][-1]
            torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))

        print(f"Epoch {epoch}/{args.max_epoch} — Train: {logs['train_acc'][-1]:.4f}, Val: {logs['val_acc'][-1]:.4f}, Time: {time.time()-start_time:.1f}s")

    print(f"Training complete. Best validation accuracy: {best_val:.4f}")

if __name__ == '__main__':
    args = get_args()
    if not hasattr(args, 'use_gnn'):
        args.use_gnn = False
    if args.use_gnn:
        args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
        args.gnn_output_dim = getattr(args, 'gnn_output_dim', 256)
    if not hasattr(args, 'latent_domain_num') or args.latent_domain_num is None:
        args.latent_domain_num = 4
    main(args)
