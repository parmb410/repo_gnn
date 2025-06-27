from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.nn import GCNConv, GATConv
from sklearn.cluster import KMeans

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

def to_device(batch, device):
    if hasattr(batch, 'to') and not isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device).float()
    else:
        raise ValueError(f"Unknown batch type: {type(batch)}")

def transform_for_gnn(x):
    if x.dim() == 4:
        if x.size(1) == 8 or x.size(1) == 200:
            x = x.squeeze(2).permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            x = x.squeeze(1).permute(0, 2, 1)
        elif x.size(3) == 8 or x.size(3) == 200:
            x = x.squeeze(2)
        elif x.size(3) == 1 and (x.size(2) == 8 or x.size(2) == 200):
            x = x.squeeze(3)
    elif x.dim() == 3:
        if x.size(1) == 8 or x.size(1) == 200:
            x = x.permute(0, 2, 1)
        elif x.size(2) == 8 or x.size(2) == 200:
            pass
    if x.dim() >= 2 and x.size(1) != 200:
        current_timesteps = x.size(1)
        if current_timesteps < 200:
            padding = 200 - current_timesteps
            x = torch.nn.functional.pad(x, (0, 0, 0, padding), "constant", 0)
        else:
            x = x[:, :200, :]
    return x

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
            self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=1)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.stack([x[batch == i].mean(0) for i in torch.unique(batch)])
        return self.classifier(x)

def init_gnn_model(args, input_dim, num_classes):
    return GNNModel(
        input_dim=input_dim,
        hidden_dim=args.gnn_hidden_dim,
        num_classes=num_classes,
        gnn_type=args.gnn_arch
    )

class Diversify(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        self.featurizer = get_fea(args)
        self.dbottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.domain_num)
        self.dclassifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.bottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.abottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(int(args.num_classes * args.latent_domain_num), args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, "label_smoothing", 0.0))
        self.lambda_cls = getattr(args, "lambda_cls", 1.0)
        self.lambda_dis = getattr(args, "lambda_dis", 0.1)
        self.explain_mode = False
        self.global_step = 0
        self.patch_skip_connection()
        
    def patch_skip_connection(self):
        device = next(self.featurizer.parameters()).device
        sample_input = torch.randn(1, *self.args.input_shape).to(device)
        with torch.no_grad():
            actual_features = self.featurizer(sample_input).shape[-1]
            print(f"Detected actual feature dimension: {actual_features}")
        for name, module in self.featurizer.named_modules():
            if isinstance(module, nn.Linear) and "skip" in name.lower():
                if module.in_features != actual_features:
                    print(f"Patching skip connection: {actual_features} features")
                    new_layer = nn.Linear(actual_features, module.out_features).to(device)
                    if module.in_features > actual_features:
                        new_layer.weight.data = module.weight.data[:, :actual_features].clone()
                        new_layer.bias.data = module.bias.data.clone()
                    elif actual_features > module.in_features:
                        new_weights = torch.randn(module.out_features, actual_features).to(device) * 0.01
                        new_weights[:, :module.in_features] = module.weight.data.clone()
                        new_layer.weight.data = new_weights
                        new_layer.bias.data = module.bias.data.clone()
                    if '.' in name:
                        parts = name.split('.')
                        parent = self.featurizer
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], new_layer)
                    else:
                        setattr(self.featurizer, name, new_layer)
                    print(f"Patched {name}: in_features={actual_features}, out_features={module.out_features}")
                return
        print("Warning: No skip connection layer found in featurizer")
        self.actual_features = actual_features

    def update_d(self, minibatch, opt):
        device = next(self.parameters()).device
        all_x1 = to_device(minibatch[0], device)
        all_d1 = minibatch[2].to(device).long()
        all_c1 = minibatch[1].to(device).long()
        n_domains = self.args.domain_num
        all_d1 = torch.clamp(all_d1, 0, n_domains - 1)
        z1 = self.dbottleneck(self.featurizer(all_x1))
        if self.explain_mode:
            z1 = z1.clone()
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        cd1 = self.dclassifier(z1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        ent_loss = Entropylogits(cd1) * self.args.lam + self.criterion(cd1, all_c1)
        loss = ent_loss + self.lambda_dis * disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()
        all_fea = []
        all_index = []
        device = next(self.parameters()).device
        with torch.no_grad():
            index_counter = 0
            for batch in loader:
                inputs = to_device(batch[0], device)
                if self.args.use_gnn:
                    inputs = transform_for_gnn(inputs)
                inputs = self.ensure_correct_dimensions(inputs)
                feas = self.dbottleneck(self.featurizer(inputs))
                all_fea.append(feas.float().cpu())
                batch_size = inputs.size(0)
                batch_indices = np.arange(index_counter, index_counter + batch_size)
                all_index.append(batch_indices)
                index_counter += batch_size
        all_fea = torch.cat(all_fea, dim=0)
        all_index = np.concatenate(all_index, axis=0)
        all_fea = all_fea / torch.norm(all_fea, p=2, dim=1, keepdim=True)
        all_fea = all_fea.float().cpu().numpy()
        K = self.args.latent_domain_num
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        pred_label = kmeans.fit_predict(all_fea)
        pred_label = np.clip(pred_label, 0, K-1)
        dataset = loader.dataset
        def get_base_dataset(ds):
            while isinstance(ds, Subset):
                ds = ds.dataset
            return ds
        base_dataset = get_base_dataset(dataset)
        if isinstance(dataset, Subset):
            current = dataset
            while isinstance(current, Subset):
                all_index = [current.indices[i] for i in all_index]
                current = current.dataset
            base_dataset = current
        if hasattr(base_dataset, 'set_labels_by_index'):
            pred_label_tensor = torch.from_numpy(pred_label).long()
            base_dataset.set_labels_by_index(pred_label_tensor, all_index, 'pdlabel')
            print(f"Set pseudo-labels on base dataset of type: {type(base_dataset).__name__}")
        else:
            print(f"Warning: Base dataset {type(base_dataset).__name__} has no set_labels_by_index method")
        counter = Counter(pred_label)
        print(f"Pseudo-domain label distribution: {dict(counter)}")
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def ensure_correct_dimensions(self, inputs):
        return inputs

    def update(self, data, opt):
        device = next(self.parameters()).device
        all_x = to_device(data[0], device)
        all_x = self.ensure_correct_dimensions(all_x)
        all_y = data[1].to(device).long()
        all_z = self.bottleneck(self.featurizer(all_x))
        if self.explain_mode:
            all_z = all_z.clone()
        self.global_step += 1
        alpha = getattr(self.args, "alpha", 1.0)
        if hasattr(self.args, "alpha_warmup") and self.args.alpha_warmup:
            total_steps = getattr(self.args, "warmup_steps", 1000)
            alpha = min(1.0, self.global_step / total_steps)
        disc_input = Adver_network.ReverseLayerF.apply(all_z, alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = data[4].to(device).long()
        disc_labels = torch.clamp(disc_labels, 0, self.args.latent_domain_num - 1)
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = self.criterion(all_preds, all_y)
        loss = self.lambda_cls * classifier_loss + self.lambda_dis * disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if self.global_step % 100 == 0:
            print(f"[Step {self.global_step}] ClassLoss={classifier_loss.item():.4f} | DiscLoss={disc_loss.item():.4f}")
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        device = next(self.parameters()).device
        all_x = to_device(minibatches[0], device)
        all_x = self.ensure_correct_dimensions(all_x)
        all_c = minibatches[1].to(device).long()
        # For this call, domain labels may be at minibatches[2]
        if len(minibatches) >= 5:
            all_d = minibatches[4].to(device).long()
        else:
            all_d = minibatches[2].to(device).long()
        n_domains = self.args.latent_domain_num
        all_d = torch.clamp(all_d, 0, n_domains-1)
        all_y = all_d * self.args.num_classes + all_c
        max_class = self.aclassifier.fc.out_features
        all_y = torch.clamp(all_y, 0, max_class-1)
        all_z = self.abottleneck(self.featurizer(all_x))
        if self.explain_mode:
            all_z = all_z.clone()
        all_preds = self.aclassifier(all_z)

        # === BEGIN SHAPE PATCH ===
        print(f"all_preds shape: {all_preds.shape}, all_y shape: {all_y.shape}")
        if all_preds.shape[0] != all_y.shape[0]:
            batch_size = all_y.shape[0]
            if all_preds.shape[0] % batch_size == 0:
                time_steps = all_preds.shape[0] // batch_size
                all_preds = all_preds.view(batch_size, time_steps, -1)
                all_preds = all_preds.mean(dim=1)  # mean over time_steps
                print(f"Reshaped/pool all_preds to {all_preds.shape}")
            else:
                raise ValueError(f"all_preds shape {all_preds.shape} and all_y shape {all_y.shape} not compatible.")
        # === END SHAPE PATCH ===

        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}

    def predict(self, x):
        x = self.ensure_correct_dimensions(x)
        device = next(self.parameters()).device
        x = to_device(x, device)
        features = self.featurizer(x)
        bottleneck_out = self.bottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.classifier(bottleneck_out)
    
    def predict1(self, x):
        x = self.ensure_correct_dimensions(x)
        device = next(self.parameters()).device
        x = to_device(x, device)
        features = self.featurizer(x)
        bottleneck_out = self.dbottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.ddiscriminator(bottleneck_out)
    
    def forward(self, batch):
        inputs = batch[0]
        inputs = self.ensure_correct_dimensions(inputs)
        device = next(self.parameters()).device
        inputs = to_device(inputs, device)
        labels = batch[1]
        preds = self.predict(inputs)
        preds = preds.float()
        labels = labels.long()
        class_loss = self.criterion(preds, labels)
        return {'class': class_loss}
    
    def explain(self, x):
        original_mode = self.explain_mode
        try:
            self.explain_mode = True
            with torch.no_grad():
                x = self.ensure_correct_dimensions(x)
                device = next(self.parameters()).device
                x = to_device(x, device)
                return self.predict(x)
        finally:
            self.explain_mode = original_mode
