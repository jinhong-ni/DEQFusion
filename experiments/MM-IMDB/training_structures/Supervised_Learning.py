
"""Implements supervised learning training procedures."""
import torch
from torch import nn
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
#import pdb

softmax = nn.Softmax()


class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
#         for o in outs:
#             print('output:',o.min().item(),o.max().item())
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
#         print('fused:',out[0].min().item(),out[0].max().item())
        self.fuseout = out
        if type(out) is tuple:
            trace = out[2]
            jac_loss = out[1]
            out = out[0]
            return self.head(out), jac_loss, trace
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)




def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True):
    """
    Handle running a simple supervised training loop.
    
    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
#         op = optimtype([p for p in model.parameters() if p.requires_grad] +
#                        additional_params, lr=lr, weight_decay=weight_decay)
        op = optimtype([
                {'params': model.encoders.parameters()},
                {'params': model.head.parameters()},
                {'params': model.fuse.parameters(), 'lr': lr*0.1}
            ], lr=lr, weight_decay=weight_decay)
#         scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=10, gamma=0.9)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            total_jac_loss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])

                else:
                    model.train()
#                     try:
                    out = model([_processinput(i).to(device)
                                    for i in j[:-1]])
#                     except:
#                         print('\nSingularity detected during training, proceeding to next batch.')
#                         continue
                jac_loss = torch.tensor([0.])
                if isinstance(out, tuple):
                    jac_loss = out[1]
                    out = out[0]
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                loss += 0.1 * jac_loss.mean()

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                try:
                    loss.backward()
                except:
                    print('\nSingularity detected during backward, proceeding to next batch.')
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print('')
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
#             scheduler.step()
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        model.eval()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        model.eval()
                        try:
                            out = model([_processinput(i).to(device)
                                        for i in j[:-1]])
                        except:
                            print('\nSingularity detected during validation')
                            continue
                    
                    jac_loss = torch.tensor([0.])
                    if isinstance(out, tuple):
                        jac_loss = out[1]
                        out = out[0]
                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    loss += 0.1 * jac_loss.mean()
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            print('')
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
    if track_complexity:
        all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        _trainprocess()


def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        abs_trace = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in j[0]], j[1]])
            else:
                out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in j[:-1]])
            jac_loss = torch.tensor([0.])
            if isinstance(out, tuple):
                abs_trace.append(out[2]['rel_trace'])
                jac_loss = out[1]
                out = out[0]
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size())-1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            else:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
#             loss += jac_loss.mean()
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
            
        import numpy as np
        import matplotlib.pyplot as plt
        trace = np.stack(abs_trace).mean(0)
        print('a!!!!!!!!!!',model.fuse.f_thres)
        plt.figure(dpi=500)
        plt.xlabel('Steps')
#         plt.ylim(-1e-2,1)
#         plt.ylim(-1,4)
        plt.ylabel('Difference norm')
        plt.title("DEQ Fusion on MM-IMDB")
        plt.plot(np.arange(len(trace)), trace)
#         plt.fill_between(np.arange(abs_diff_trace.shape[1]), min_abs_diff, max_abs_diff,
#                          facecolor="blue", # The fill color
#                          color='blue',       # The outline color
#                          alpha=0.2) 
        plt.savefig('mm-imdb.pdf')
        np.save('mm-imdb.npy', trace)
                
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}



def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=True):
    """
    Handle getting test results for a simple supervised training loop.
    
    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there   
    """
    if no_robust:
        def _testprocess():
            single_test(model, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(_testprocess, [model])
        return

    def _testprocess():
        single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[
                    0]][0], is_packed, criterion, task, auprc, input_to_float)
    all_in_one_test(_testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)
