import torch
from utils.metrics import evaluate_cls, evaluate_mcls, evaluate_reg
import json
from utils import unbatch
from reprint import output
import math
import os
import sys

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, lrate, min_lrate, wdecay, betas, eps, amsgrad, clip, steps_per_epoch, num_epochs,
                 total_iters,
                 warmup_iters=2000, lr_decay_iters=None, schedule_lr=True, regression_weight=1,
                 classification_weight=1, multiclassification_weight=1, evaluate_metric='rmse',
                 result_path='', runid=0, device='cuda:0', skip_test_during_train=False,
                 finetune_modules=None):

        self.model = model
        self.model.to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay=wdecay, learning_rate=lrate,
                                                         betas=betas, eps=eps, amsgrad=amsgrad)  # amsgrad是Adam 优化器的一个变种
        if finetune_modules is not None:
            self.optimizer = self.model.freeze_backbone_optimizers(finetune_modules, weight_decay=wdecay,
                                                                   learning_rate=lrate,
                                                                   betas=betas, eps=eps, amsgrad=amsgrad)
            print('freezing backbone and now training only the finetune modules...')

        self.clip = clip  # 是否要进行梯度裁剪
        self.regression_loss = missing_mse_loss  # MSE损失
        self.classification_loss = torch.nn.BCEWithLogitsLoss()  # 分类损失
        self.mclassification_loss = missing_ce_loss  # 多分类损失

        self.num_epochs = num_epochs

        self.result_path = result_path
        self.runid = runid
        self.device = device
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.multiclassification_weight = multiclassification_weight
        self.evaluate_metric = evaluate_metric  # rmse
        self.skip_test_during_train = skip_test_during_train

        self.schedule_lr = schedule_lr
        if total_iters:
            self.total_iters = total_iters
        else:
            self.total_iters = num_epochs * steps_per_epoch

        self.lrate = lrate
        self.min_lrate = min_lrate
        self.warmup_iters = warmup_iters
        if lr_decay_iters is None:
            self.lr_decay_iters = self.total_iters
        else:
            self.lr_decay_iters = lr_decay_iters
        self.con_weight = 0.2

    def train_epoch(self, train_loader, val_loader=None, test_loader=None, evaluate_epoch=1):
        if self.evaluate_metric in ['rmse', 'mse', 'mae']:
            best_result = float('inf')  # 设置初始为无穷大
            best_result_test = float('inf')

        else:
            best_result = float('-inf')  # 设置初始为无穷小
            best_result_test = float('-inf')
        pbar = tqdm(total=self.total_iters, desc='training')
        iter_num = 0
        val_str = ''
        test_str = ''
        with output(initial_len=11, interval=0) as output_lines:
            for epoch in range(1, self.num_epochs + 1):
                running_reg_loss = 0
                running_cls_loss = 0
                running_mcls_loss = 0
                running_spectral_loss = 0
                running_ortho_loss = 0
                running_cluster_loss = 0
                self.model.train()

                for data in train_loader:
                    if self.schedule_lr:  # 是否启用学习率调度
                        curr_lr_rate = self.get_lr(iter_num)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = curr_lr_rate
                    else:
                        curr_lr_rate = self.lrate

                    self.optimizer.zero_grad()

                    data = data.to(self.device)
                    reg_pred, sp_loss, o_loss, cl_loss, pos_loss, mi_loss = self.model(
                        # Molecule
                        mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                        atom_edge_index=data.mol_edge_index, clique_x=data.clique_x,
                        clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                        # Protein
                        residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                        # residue_degree=data.prot_degree,
                        residue_edge_index=data.prot_edge_index,
                        residue_edge_weight=data.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch,
                        clique_batch=data.clique_x_batch,
                        mode='train'
                    )
                    ## Loss compute
                    cls_loss = 0  # 分类损失
                    mcls_loss = 0  # 多分类损失
                    reg_loss = 0  # 回归损失

                    loss_val = torch.tensor(0.).to(self.device)
                    loss_val += sp_loss
                    loss_val += o_loss
                    loss_val += cl_loss

                    # 将这些损失转换为可打印的标量值
                    sp_loss = sp_loss.item()
                    o_loss = o_loss.item()
                    cl_loss = cl_loss.item()

                    reg_pred = reg_pred.squeeze()  # 去掉张量中为1的维度
                    reg_y = data.reg_y.squeeze()  # 去掉张量中为1的维度
                    reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight  # 计算MSE损失
                    loss_val += reg_loss  # 累加损失
                    reg_loss = reg_loss.item()  # 提取标量损失值
                    loss_val = loss_val * (1 - self.con_weight) + (self.con_weight / 2) * pos_loss + (
                                self.con_weight / 2) * mi_loss

                    loss_val.backward()  # 梯度更新

                    if self.clip is not None:  # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    self.model.temperature_clamp()  # 将模型中的某些参数值限制在合理的范围内。
                    running_reg_loss += reg_loss
                    running_cls_loss += cls_loss
                    running_mcls_loss += mcls_loss

                    running_spectral_loss += sp_loss
                    running_ortho_loss += o_loss
                    running_cluster_loss += cl_loss
                    pbar.update(1)  # 更新进度条 pbar
                    iter_num += 1
                    # 结束for循环--------------------------

                train_reg_loss = running_reg_loss / len(train_loader)
                train_cls_loss = running_cls_loss / len(train_loader)
                train_mcls_loss = running_mcls_loss / len(train_loader)
                train_spectral_loss = running_spectral_loss / len(train_loader)
                train_ortho_loss = running_ortho_loss / len(train_loader)
                train_cluster_loss = running_cluster_loss / len(train_loader)

                train_str1 = f"Train MSE Loss: {train_reg_loss:.4f}, Train CLS Loss: {train_cls_loss:.4f}, Train MCLS Loss: {train_mcls_loss:.4f}"
                train_str2 = f"Protein Cluster Region Loss - Ortho Loss: {train_ortho_loss:.4f}, Cluster Loss: {train_cluster_loss:.4f}"

                if epoch % evaluate_epoch == 0 and val_loader is not None:
                    val_result = self.eval(val_loader)
                    val_result = {k: round(v, 4) for k, v in val_result.items()}
                    val_str = f'Validation Results: ' + json.dumps(val_result, indent=4, sort_keys=True)
                    if self.evaluate_metric in ['rmse', 'mse', 'mae']:
                        if val_result[self.evaluate_metric] < best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False
                    else:
                        if val_result[self.evaluate_metric] > best_result:
                            better_than_previous = True
                        else:
                            better_than_previous = False

                    if better_than_previous:
                        best_result = val_result[self.evaluate_metric]
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.result_path, 'save_model_seed{}'.format(self.runid), 'model.pt'))


                    test_result = self.eval(test_loader)

                    if self.evaluate_metric in ['rmse', 'mse', 'mae']:
                        if test_result[self.evaluate_metric] < best_result_test:
                            better_than_previous = True
                        else:
                            better_than_previous = False
                    else:
                        if test_result[self.evaluate_metric] > best_result_test:
                            better_than_previous = True
                        else:
                            better_than_previous = False

                    if better_than_previous:
                        best_result_test = test_result[self.evaluate_metric]
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.result_path, 'save_model_seed{}'.format(self.runid), 'model_test.pt'))

                test_result = {k: round(v, 4) for k, v in test_result.items()}
                test_str = f'Test Results: ' + json.dumps(test_result, indent=4, sort_keys=True)
                output_lines[0] = ' ' * 30
                output_lines[1] = ' ' * 30
                output_lines[2] = '-' * 40
                output_lines[3] = f'Epoch {epoch:03d} with LR {curr_lr_rate:.6f}: Model Results'
                output_lines[4] = '-' * 40
                output_lines[5] = train_str1
                output_lines[6] = train_str2
                output_lines[7] = ' ' * 30
                output_lines[8] = val_str
                output_lines[9] = ' ' * 30
                output_lines[10] = test_str

                with open(self.result_path + '/full_result-{}.txt'.format(self.runid), 'a+') as f:
                    f.write('-' * 30 + f'\nEpoch: {epoch:03d} - Model Results\n' + '-' * 30 + '\n')
                    f.write(train_str1 + '\n')
                    f.write(train_str2 + '\n')
                    f.write(val_str + '\n')
                    f.write(test_str + '\n')

    def eval(self, data_loader):
        reg_preds = []
        reg_truths = []

        running_reg_loss = 0
        running_cls_loss = 0
        running_mcls_loss = 0

        running_spectral_loss = 0
        running_ortho_loss = 0
        running_cluster_loss = 0

        self.model.eval()
        eval_result = {}
        with torch.no_grad():
            for data in tqdm(data_loader, leave=False, desc='evaluating'):
                data = data.to(self.device)
                reg_pred, sp_loss, o_loss, cl_loss, _, _ = self.model(
                    # Molecule
                    mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                    atom_edge_index=data.mol_edge_index, clique_x=data.clique_x,
                    clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                    # Protein
                    residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                    residue_edge_index=data.prot_edge_index,
                    residue_edge_weight=data.prot_edge_weight,
                    # residue_degree=data.prot_degree,
                    # Mol-Protein Interaction batch
                    mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch,
                    mode='valid_test'
                )
                ## Loss compute
                cls_loss = 0
                mcls_loss = 0
                reg_loss = 0

                loss_val = 0
                loss_val += sp_loss
                loss_val += o_loss
                loss_val += cl_loss
                sp_loss = sp_loss.item()
                o_loss = o_loss.item()
                cl_loss = cl_loss.item()

                if reg_pred is not None:
                    reg_pred = reg_pred.squeeze().reshape(-1)
                    reg_y = data.reg_y.squeeze().reshape(-1)
                    reg_loss = self.regression_loss(reg_pred, reg_y) * self.regression_weight
                    loss_val += reg_loss
                    reg_loss = reg_loss.item()  # 将张量转换为标量值
                    reg_preds.append(reg_pred)
                    reg_truths.append(reg_y)

                running_reg_loss += reg_loss
                running_cls_loss += cls_loss
                running_mcls_loss += mcls_loss
                running_spectral_loss += sp_loss
                running_ortho_loss += o_loss
                running_cluster_loss += cl_loss

            eval_reg_loss = running_reg_loss / len(data_loader)
            eval_spectral_loss = running_spectral_loss / len(data_loader)
            eval_ortho_loss = running_ortho_loss / len(data_loader)
            eval_cluster_loss = running_cluster_loss / len(data_loader)

            eval_result['regression_loss'] = eval_reg_loss
            eval_result['spectral_loss'] = eval_spectral_loss
            eval_result['ortho_loss'] = eval_ortho_loss
            eval_result['cluster_loss'] = eval_cluster_loss

        if len(reg_truths) > 0:
            reg_preds = torch.cat(reg_preds).detach().cpu().numpy()
            reg_truths = torch.cat(reg_truths).detach().cpu().numpy()
            eval_reg_result = evaluate_reg(reg_truths, reg_preds)
            eval_result.update(eval_reg_result)

        return eval_result



    def get_lr(self, iter):  # 带有线性预热（linear warmup）和余弦衰减（cosine decay）的学习率调度策略
        # 1) linear warmup for warmup_iters steps
        if iter < self.warmup_iters:
            return self.lrate * iter / self.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > self.lr_decay_iters:
            return self.min_lrate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return self.min_lrate + coeff * (self.lrate - self.min_lrate)


def masked_mse_loss(pred, true):
    mask = ~torch.isnan(true)
    pred = torch.masked_select(pred, mask)
    true = torch.masked_select(true, mask)
    mse_val = torch.mean((true - pred) ** 2)

    return mse_val


def store_attention_result(attention_dict, keys, reg_tuples=None, cls_tuples=None):
    interpret_dict = {}

    unbatched_residue_score = unbatch(attention_dict['residue_final_score'], attention_dict['protein_residue_index'])
    unbatched_atom_score = unbatch(attention_dict['atom_final_score'], attention_dict['drug_atom_index'])

    unbatched_residue_layer_score = unbatch(attention_dict['residue_layer_scores'],
                                            attention_dict['protein_residue_index'])
    unbatched_clique_layer_score = unbatch(attention_dict['clique_layer_scores'], attention_dict['drug_clique_index'])

    for idx, key in enumerate(keys):
        interpret_dict[key] = {
            'residue_score': unbatched_residue_score[idx].detach().cpu().numpy(),
            'atom_score': unbatched_atom_score[idx].detach().cpu().numpy(),
            'residue_layer': unbatched_residue_layer_score[idx].detach().cpu().numpy(),
            'clique_layer': unbatched_clique_layer_score[idx].detach().cpu().numpy(),
            'mol_feature': attention_dict['mol_feature'][idx].detach().cpu().numpy(),
            'prot_feature': attention_dict['prot_feature'][idx].detach().cpu().numpy(),
            'interaction_fingerprint': attention_dict['interaction_fingerprint'][idx].detach().cpu().numpy(),
        }
        if cls_tuples:
            interpret_dict[key]['classification_truth'] = cls_tuples[idx][0].item()
            interpret_dict[key]['classification_prediction'] = cls_tuples[idx][1].item()
        if reg_tuples:
            interpret_dict[key]['regression_truth'] = reg_tuples[idx][0].item()
            interpret_dict[key]['regression_prediction'] = reg_tuples[idx][1].item()

    return interpret_dict


def missing_mse_loss(pred, true, threshold=5.000):
    loss = torch.tensor(0.).to(pred.device)

    ## true labels available
    if (~torch.isnan(true)).any():
        real_mask = ~torch.isnan(true)
        real_pred = torch.masked_select(pred, real_mask)
        real_true = torch.masked_select(true, real_mask)
        loss += torch.mean((real_true - real_pred) ** 2)

    return loss


def missing_ce_loss(pred, true, negative_cls=1):
    mclass_criterion = torch.nn.CrossEntropyLoss()
    negative_class_criterion = torch.nn.BCELoss()
    loss = torch.tensor(0.).to(pred.device)
    counter = 0

    if (~torch.isnan(true)).any():
        real_mask = ~torch.isnan(true)
        real_pred = pred[real_mask]
        real_true = true[real_mask]

        ## unknown
        unknown_mask = torch.where(real_true == 1000)[0]
        unknown_pred = real_pred[unknown_mask]
        if len(unknown_pred) > 0:
            unknown_pred = unknown_pred.softmax(dim=-1)
            ## take binder class (agonist and antagonist) only ##
            positive_cls = torch.ones(unknown_pred.shape[-1]).bool()
            positive_cls[negative_cls] = False
            positive_cls = positive_cls.to(pred.device)
            unknown_pred = unknown_pred[:, positive_cls].sum(dim=-1)
            ## take binder class (agonist and antagonist) only ##
            unknown_true = torch.ones(unknown_pred.size(0)).float().to(pred.device)  ## all of them are positives
            unknown_pred = torch.where(torch.isnan(unknown_pred), torch.zeros_like(unknown_pred), unknown_pred).clamp(0,
                                                                                                                      1)

            loss += negative_class_criterion(unknown_pred, unknown_true)
            counter += 1

        ## known values
        known_mask = torch.where(real_true != 1000)[0]
        known_pred = real_pred[known_mask]
        known_true = real_true[known_mask]
        if len(known_pred) > 0:
            known_true = known_true.long()
            loss += mclass_criterion(known_pred, known_true)
            counter += 1

    return loss