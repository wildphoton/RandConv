#!/usr/bin/env python
"""
Train classfication network with random convolution layer
Created by zhenlinxu on 11/23/2019
"""
import os
import sys
sys.path.append(os.path.abspath(''))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms

from tqdm import tqdm, trange

# from models import *
from lib.networks import RandConvModule
from lib.utils.average_meter import AverageMeter
from lib.utils.metrics import accuracy
os.environ['TORCH_HOME'] = os.path.realpath('lib/networks/')  # where the pytorch pretrained model is saved

def add_basic_args(parser):
    parser.add_argument('--tag', '-tag', type=str, default='', help='extra tag for the experiment')
    parser.add_argument('--lr', '-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--resume', '-resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu_ids', '-g', type=int, default=1, help='ids of GPUs to use')
    parser.add_argument('--n_epoch', '-ne', type=int, default=100, help='number of trainning epochs')

    # set training iterations when epoch does not exits
    parser.add_argument('--n_iter', '-ni', type=int, help='number of total trainning iterations')
    parser.add_argument('--val_iter', '-vi', type=int, default=100, help='number of training iterations between two validations')

    parser.add_argument('--batch_size', '-bsz', type=int, default=32, help='ids of GPUs to use')
    parser.add_argument('--rand_seed', '-rs', type=int,  help='random seed')
    parser.add_argument('--data_name', '-dn', type=str, help='name of data')
    parser.add_argument('--net', '-net', type=str, default='alexnet',  help='network')
    parser.add_argument('--pretrained', '-pt', action='store_true',  help='use pretrained network')
    parser.add_argument('--test', '-test', action='store_true', help='run testing only')

    parser.add_argument('--grey', '-gr', action='store_true',
                        help='using gray scale images')
    parser.add_argument('--color_jitter', '-jitter', action='store_true')
    parser.add_argument('--multi_aug', '-ma', action='store_true', help='strong data augmentations')
    parser.add_argument('--LoG', '-LoG', action='store_true', help='use Laplacian of Gaussian for data augmentation')

    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--nesterov', '-nest', action='store_true', help='use nesterov momentum')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', '-mmt', default=0.9, type=float, help='momentum')

    parser.add_argument('--scheduler', '-sch', type=str, default='', help='type of lr scheduler, StepLR/MultiStepLR/CosLR')
    parser.add_argument('--step_size', '-stp', type=int, default=30, help='fixed step size for StepLR')
    parser.add_argument('--milestones', '-milestones', type=int, nargs='+', help='milestone for MultiStepLR')
    parser.add_argument('--gamma', '-gm', type=float,  default=0.2, help='reduce rate for step scheduler')
    parser.add_argument('--power', '-power', default=0.9, help='power for poly scheduler')

    parser.add_argument('--freeze_feature', '-ff', action='store_true', help='freeze the feature extractor during training')

    parser.add_argument('--save_embedding', '-se', action='store_true', help='save embedding during testing for visualization')
    parser.add_argument('--test_corrupted', '-crpt', action='store_true',
                        help='test robustness with corrupted data')

    parser.add_argument('--image_size', '-isz', type=int, default=-1,
                        help='resize input image size, -1 means keep original size')
    parser.add_argument('--n_classes', '-nc', type=int, default=10,
                            help='number of classes')

def add_rand_layer_args(parser):
    parser.add_argument('--rand_conv', '-rc', action='store_true', help='use random layers')
    parser.add_argument('--channel_size', '-chs', type=int, default=3,
                        help='Number of output channel size  random layers, '
                        )
    parser.add_argument('--kernel_size', '-ks', type=int, default=[3,], nargs='+',
                        help='kernal size for random layer, could be multiple kernels for multiscale mode')

    parser.add_argument('--rand_bias', '-rb', action='store_true',
                        help='add random bias in convolution layer')

    parser.add_argument('--distribution', '-db', type=str, default='kaiming_normal',
                        help='distribution of random sampling')

    parser.add_argument('--clamp_output', '-clamp', action='store_true',
                        help='clamp value range of randconv outputs to a range (as in original image)'
                        )

    parser.add_argument('--mixing', '-mix', action='store_true',
                        help='mix the output of rand conv layer with the original input')

    parser.add_argument('--identity_prob', '-idp', type=float, default=0.0,
                        help='the probability that the rand conv is a identity map, '
                             'in this case, the output and input must have the same channel number')

    parser.add_argument('--multi_scale', '-ms', type=str, nargs='+',
                        help='multiscale settings, e.g. \'3-3\' means kernel size 3 with output channel size 3')

    parser.add_argument('--rand_freq', '-rf', type=int, default=1,
                        help='frequency of randomize weights of random layers (every n steps)')

    parser.add_argument('--train-all', '-ta', action='store_true',
                        help='train all random layers, use for ablation study when the network is modified')

    parser.add_argument('--consistency_loss', '-cl', action='store_true',
                        help='use invariant loss to enforce similar predictionso on different augmentation of the same input')
    parser.add_argument('--consistency_loss_w', '-clw', type=float, default=1.0,
                        help='weight for invariant loss')
    parser.add_argument('--augmix', '-am', action='store_true',
                        help='aug_mix mode, only the raw data is used to compute classfication loss')

    parser.add_argument('--n_val', '-nv', type=int, default=1,
                        help='repeat validation with different randconv')
    parser.add_argument('--val_with_rand', '-vwr', action='store_true',
                        help='validation with random conv;'
                        )

    parser.add_argument('--test_latest', '-tl', action='store_true',
                        help='test the last saved model instead of the best one')
    parser.add_argument('--test_target', '-tt', action='store_true',
                        help='test the best model on target domain')



def get_exp_name(args):
    exp_name = "".join([
        args.net,
        '-pretrained' if args.pretrained else '',
        '-freeze_feature' if args.freeze_feature else '',
        '-MultiAug' if args.multi_aug else '',
        '-colorjitter' if args.color_jitter else '',
        '-grey' if args.grey else '',
        '-LoG_sz{}_sigma{}_p{}'.format(args.LoG_size, args.LoG_sigma, args.LoG_p) if args.LoG else '',
        '-clampOutput' if args.clamp_output and args.rand_conv else '',
        '-randConv' if args.rand_conv else '',
        '-ch{}'.format(args.channel_size) if args.rand_conv else '',
        '-{}'.format(args.distribution) if args.distribution and args.rand_conv else '',
        '-kz{}'.format('_'.join(str(k) for k in args.kernel_size)) if args.kernel_size and args.rand_conv else '',

        '-randbias'.format(args.rand_bias) if args.rand_bias and args.rand_conv else '',
        '-mixing' if args.mixing and args.rand_conv else '',
        '-idprob_{}'.format(args.identity_prob) if args.identity_prob > 0 and args.rand_conv else '',
        '-freq{}'.format(args.rand_freq) if (args.rand_conv and not args.train_all) else '',
        '-{}cons_{}'.format('augmix-' if args.augmix else '', args.consistency_loss_w)
        if args.consistency_loss and args.rand_conv and not args.train_all else '',
        '-val_rand{}'.format(args.n_val)
        if args.val_with_rand and args.rand_conv else '',
        '-trainall' if args.train_all else '',
        '-lr{}'.format(args.lr),
        '-batch{}'.format(args.batch_size),
        '-SGD-{}mom{}-wd{}'.format('nesterov' if args.nesterov else '', args.momentum, args.weight_decay) if args.SGD else '',
        '-{}Schd_step{}_gamma{}'.format(args.scheduler, args.step_size, args.gamma) if args.scheduler == 'StepLR' else '',
        '-{}Schd_{}_gamma{}'.format(args.scheduler, '_'.join([str(i) for i in args.milestones]), args.gamma) if args.scheduler == 'MultiStepLR' else '',
        '-{}Schd'.format(args.scheduler) if args.scheduler == 'CosLR' else '',
        '-{}ep'.format(args.n_epoch) if not args.n_iter else '-{}iters'.format(args.n_iter),
        args.tag
    ])
    return exp_name

def get_random_module(net, args, data_mean, data_std):
    return RandConvModule(net,
                          in_channels=3,
                          out_channels=args.channel_size,
                          kernel_size=args.kernel_size,
                          mixing=args.mixing,
                          identity_prob=args.identity_prob,
                          rand_bias=args.rand_bias,
                          distribution=args.distribution,
                          data_mean=data_mean,
                          data_std=data_std,
                          clamp_output=args.clamp_output,
                          )

class RandCNN:
    def __init__(self, args):
        hostname = os.uname()[1]
        if 'biag' in hostname or 'unc.edu' not in hostname:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_ids) if type(args.gpu_ids) is list else str(args.gpu_ids)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            cudnn.benchmark = True

        if args.n_iter:
            args.n_epoch = args.n_iter // args.val_iter

        self.args = args

        self.exp_name = get_exp_name(self.args)

        print("{} experiments with {} as source domain {}".format(self.args.data_name, self.args.source, self.exp_name))
        self.set_path()
        self.criterion = nn.CrossEntropyLoss()
        if self.args.consistency_loss:
            self.invariant_criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            self.invariant_criterion = None

        self.writer = SummaryWriter(os.path.join(self.ckpoint_folder, 'seed{}'.format(self.args.rand_seed)))
        self.metric_name = 'acc'
        self.metric = accuracy

        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        self.gray = transforms.RandomGrayscale()


    def multi_augment(self, input):
        if self.args.multi_aug:
            input = self.dewhiten(input)
            input = self.jitter(input)
            input = self.gray(input)
            return self.whiten(input).contiguous()
        else:
            return input

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean


    def set_path(self):
        self.ckpoint_folder = os.path.join('checkpoints', self.args.data_name, self.exp_name, self.args.source)
        self.best_ckpoint = os.path.join(self.ckpoint_folder, 'best{}.ckpt.pth'.format('_seed{}'.format(self.args.rand_seed) if self.args.rand_seed is not None else ''))
        self.best_target_ckpoint = os.path.join(self.ckpoint_folder, 'best_target{}.ckpt.pth'.format('_seed{}'.format(self.args.rand_seed) if self.args.rand_seed is not None else ''))
        self.last_ckpoint = os.path.join(self.ckpoint_folder, 'last{}.ckpt.pth'.format('_seed{}'.format(self.args.rand_seed) if self.args.rand_seed is not None else ''))

        if self.args.test_corrupted:
            log_file_name = "log_corrupt"
        else:
            log_file_name = 'log'

        log_file_name += '{}.txt'.format("".join(["_seed{}".format(self.args.rand_seed) if self.args.rand_seed is not None else '',
                                                  '_target' if self.args.test_target else ''])
                                         )

        self.log_path = os.path.join(self.ckpoint_folder, log_file_name)
        print('log path', self.log_path)

    def set_optimizer_and_scheduler(self, paras, lr, SGD=False, momentum=0.9, weight_decay=5e-4, nesterov=False,
                                    scheduler_name='', step_size=20, gamma=0.1, milestones=(10,20), n_epoch=30, power=2):
        # only update non-random layers
        if SGD:
            print("Using SGD optimizer")
            optimizer = optim.SGD(paras, lr=lr, momentum=momentum,
                                  weight_decay=weight_decay, nesterov=nesterov)
        else:
            print("Using Adam optimizer")
            optimizer = optim.Adam(paras, lr=lr)

        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
                                                  gamma=gamma)
        elif scheduler_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                       gamma=gamma)
        elif scheduler_name == 'CosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_steps_per_epoch * n_epoch)
        elif not scheduler_name:
            scheduler = None
        else:
            raise NotImplementedError()

        return optimizer, scheduler


    def train(self, net, trainloaders, validloaders, testloaders=None, data_mean=None, data_std=None, net_paras=None):
        """
        Training a classfication CNN with random layers

        :param net: base model
        :param trainloaders:
        :param validloaders:
        :param testloaders:
        :param data_mean: mean of dataset (a vector of 3 for color images)
        :param data_std: std of dataset (a vector of 3 for color images)
        :param net_paras: optional, the paprameter of base model to be optimized. Use it when customized training needed
        :return:
        """
        self.data_mean = torch.tensor(data_mean).reshape(3, 1, 1).to(self.device)
        self.data_std = torch.tensor(data_std).reshape(3, 1, 1).to(self.device)
        self.best_metric = 0  # best valid accuracy
        self.best_target_metric = 0  # best valid accuracy on target domain
        self.current_metric = 0
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # get the random conv layers and trainable parameters
        self.paras = []

        net = net.to(self.device)
        if self.args.freeze_feature:
            self.paras += list(net.classifier.parameters())
        else:
            if net_paras is not None:
                self.paras += list(net_paras)
            else:
                self.paras += [{'params': net.parameters()}]

        if self.args.rand_conv:
            print("\n=========Set Up Rand layers=========")
            self.rand_module = get_random_module(net, self.args, data_mean, data_std)
            self.rand_module.to(self.device)
        else:
            self.rand_module = None

        if self.args.n_iter:
            self.n_steps_per_epoch = self.args.val_iter
        else:
            self.n_steps_per_epoch = max([len(loader) for loader in trainloaders])


        if not self.args.test:
            print("\n=========Training with rand layers=========")
            optimizer, scheduler = self.set_optimizer_and_scheduler(
                self.paras, lr=self.args.lr, SGD=self.args.SGD, momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                nesterov=self.args.nesterov, scheduler_name=self.args.scheduler, step_size=self.args.step_size,
                gamma=self.args.gamma, milestones=self.args.milestones, n_epoch=self.args.n_epoch, power=self.args.power)


            for epoch in trange(start_epoch, self.args.n_epoch, leave=True):
                self.train_one_epoch(epoch, net, self.device, trainloaders, self.criterion, optimizer, self.args,
                                          self.invariant_criterion, scheduler)

                #  validation
                source_metric, target_metric = self.validate(epoch, net, self.device, validloaders,
                                                self.args.n_val if self.args.val_with_rand else 1,  optimizer)

                print("Source {}: {}, Best Source {}: {}, target {}: {}, Best target {}: {}".format(self.metric_name, source_metric, self.metric_name, self.best_metric, self.metric_name, target_metric, self.metric_name, self.best_target_metric))
                if scheduler is not None and self.args.scheduler != 'CosLR':
                    scheduler.step()

        self.print_now()


    def run_testing(self, net, testloaders):
        print("\n=========Testing=========")
        self.resume_model(net, test_latest=self.args.test_latest, test_target=self.args.test_target)

        self.test(net, testloaders)
        self.print_now()

    # Training one epoch
    # @staticmethod
    def train_one_epoch(self, epoch, net, device, trainloaders, criterion, optimizer, args, invariant_criterion=None,
                        scheduler=None):
        net.train()
        train_loss = 0
        train_inv_loss = 0


        if not isinstance(trainloaders, dict) and not isinstance(trainloaders, list):
            trainloaders = [trainloaders]

        dataiters = [iter(dataloader) for dataloader in trainloaders]

        metric_meter = AverageMeter(self.metric_name, ':6.2f')

        with trange(self.n_steps_per_epoch, desc='Epoch[{}/{}]'.format(epoch, self.args.n_epoch)) as t:
            for batch_idx in t:
                if not self.args.train_all and self.rand_module is not None \
                        and ((batch_idx+1) % args.rand_freq == 0 or batch_idx == 0):
                    self.rand_module.randomize()

                loss = 0.0
                inv_loss = 0.0
                for d in range(len(trainloaders)):
                    try:
                        inputs, targets = next(dataiters[d])
                    except:
                        dataiters[d] = iter(trainloaders[d])
                        inputs, targets = next(dataiters[d])
                    inputs, targets = inputs.to(device), targets.to(device)

                    if self.rand_module is None or (self.args.consistency_loss and self.args.augmix):
                        inputs = self.multi_augment(inputs)
                    else:
                        inputs = self.multi_augment(self.rand_module(inputs))
                    outputs = net(inputs)
                    loss += criterion(outputs, targets)

                    metric = self.metric(outputs, targets)
                    metric_meter.update(metric, outputs.size(0))

                    if not self.args.train_all and self.rand_module is not None and invariant_criterion is not None:
                        # if self.args.augmix:
                        self.rand_module.randomize()
                        outputs1 = net(self.multi_augment(self.rand_module(inputs)))
                        self.rand_module.randomize()
                        outputs2 = net(self.multi_augment(self.rand_module(inputs)))

                        if args.consistency_loss:
                            p_clean, p_aug1, p_aug2 = F.softmax(
                                outputs, dim=1), F.softmax(
                                outputs1, dim=1), F.softmax(
                                outputs2, dim=1)
                            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                            inv_loss += (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                         F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                         F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.



                # get loss over dataloaders
                if not self.args.train_all and self.rand_module is not None and invariant_criterion is not None:
                    train_loss += loss.item()
                    train_inv_loss += inv_loss.item()
                    loss += inv_loss * args.consistency_loss_w
                else:
                    train_loss += loss.item()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and self.args.scheduler == 'CosLR':
                    scheduler.step()

                global_step = epoch*self.n_steps_per_epoch + batch_idx

                if global_step % 100 == 0:
                    self.writer.add_scalar('Loss/train_cls', train_loss / (batch_idx + 1), global_step)
                    self.writer.add_scalar('{}/train'.format(self.metric_name), metric_meter.avg, global_step)

                if args.consistency_loss and self.rand_module is not None:
                    t.set_postfix_str('loss: %.3f/%.3f | %s: %.3f%% (%d/%d)'
                                      % (train_loss / (batch_idx + 1), train_inv_loss / (batch_idx + 1), self.metric_name,  metric_meter.avg*100, metric_meter.sum, metric_meter.count))

                    if global_step % 100 == 0:
                        self.writer.add_scalar('Loss/train_inv', train_inv_loss / (batch_idx + 1), global_step)

                else:
                    t.set_postfix_str('loss: %.3f | %s: %.3f%% (%d/%d)'
                                      % (train_loss / (batch_idx + 1), self.metric_name,  metric_meter.avg*100, metric_meter.sum, metric_meter.count))



    def infer(self, net, device, testloader, criterion=None, with_rand=False, n_eval=1, name='', label_collection=None):
        """
        base function for validation/testing with options of repeat multiple runs
        """
        net.eval()
        test_loss = 0
        metric_meter = AverageMeter(self.metric_name, ':6.2f')

        if self.rand_module is None:
            n_eval = 1

        loader_iter = iter(testloader)
        with torch.no_grad():
            with trange(len(testloader), desc='Testing {}'.format(name if name else ''), leave=True) as t:
                for batch_idx in t:
                    inputs, targets = next(loader_iter)
                    if type(label_collection) is list:
                        label_collection.append(targets)

                    inputs, targets = inputs.to(device), targets.to(device)

                    pred_batchs = []
                    for i in range(n_eval):
                        if with_rand and self.rand_module is not None:
                            self.rand_module.randomize()
                            outputs = net(self.multi_augment(self.rand_module(inputs)))

                        else:
                            outputs = net(inputs)

                        _, predicted = outputs.max(1)
                        pred_batchs.append(predicted)

                        if criterion is not None:
                            loss = criterion(outputs, targets)/n_eval
                            test_loss += loss.item()

                    # get the avg validation metric of n_eval runs
                    metric = 0
                    for pred_ in pred_batchs:
                        metric += self.metric(pred_, targets)
                    metric /= len(pred_batchs)

                    metric_meter.update(metric, inputs.size(0))

                    t.set_postfix_str('Loss: %.3f | %s: %.3f%% (%d/%d)'
                                      % (test_loss / (batch_idx + 1), self.metric_name,  metric_meter.avg*100, metric_meter.sum, metric_meter.count))

        return metric_meter.avg*100, test_loss / (batch_idx + 1)

    def validate(self, epoch, net, device, validloaders, n_eval=1, optimizer=None):
        """"""
        if not isinstance(validloaders, dict) and not isinstance(validloaders, list):
            validloaders = [validloaders]

        net.eval()
        source_metrics = []
        target_metrics = []
        global_step = (epoch+1) * self.n_steps_per_epoch

        with torch.no_grad():
            for i, loader in enumerate(validloaders):
                if type(validloaders) is dict:
                    name = loader
                    loader = validloaders[name]
                else:
                    name = None

                metric_temp, loss_temp = self.infer(net, device, loader, self.criterion, with_rand=self.args.val_with_rand,
                                                     n_eval=n_eval, name=name)

                if name is not None and name in self.args.source:
                    source_metrics.append(metric_temp)
                if name is not None and name not in self.args.source:
                    target_metrics.append(metric_temp)
                if name is None:
                    source_metrics.append(metric_temp)
                self.writer.add_scalar('{}/valid_{}'.format(self.metric_name, name if name is not None else i), metric_temp, global_step)

        # Save checkpoint.
        if source_metrics:
            source_metric = np.mean(source_metrics)
        else:
            source_metric = 0

        if source_metrics:
            self.writer.add_scalar('{}/valid_avg'.format(self.metric_name), source_metric, global_step)

        print('Saving..')

        if not os.path.isdir(self.ckpoint_folder):
            os.makedirs(self.ckpoint_folder)

        state = {
            'net': net.state_dict(),
            self.metric_name: source_metric,
            'epoch': epoch,
        }

        if self.rand_module is not None:
            state['rand_module'] = self.rand_module.state_dict()

        if source_metric > self.best_metric:
            print('Best validation {}!'.format(self.metric_name))
            self.best_metric = source_metric
            torch.save(state, self.best_ckpoint)

        if target_metrics:
            target_metric = np.mean(target_metrics)
            if target_metric > self.best_target_metric:
                self.best_target_metric = target_metric
                state[self.metric_name] = target_metric
                print('Best validation {} on target domain!'.format(self.metric_name))
                torch.save(state, self.best_target_ckpoint)
        else:
            target_metric = 0

        if (epoch + 1) % 10 == 0:
            if optimizer:
                state['optimizer'] = optimizer.state_dict()
            torch.save(state, self.last_ckpoint)

        return source_metric, target_metric

    def test(self, net, testloaders):

        metrics = {name: 0 for name in testloaders.keys()}
        with tqdm(testloaders.items(), leave=False, desc='Domains: ') as d_iter:
            for name, loader in d_iter:

                if self.args.save_embedding:
                    embeddings = []
                    labels = []
                    def feature_hook(module, input, output):
                        embeddings.append(input[0].detach().cpu())
                    h = net.classifier[-1].register_forward_hook(feature_hook)
                else:
                    labels = None

                metric, _ = self.infer(net, self.device, loader, name=name, label_collection=labels)
                metrics[name] = metric
                d_iter.set_postfix_str("Domain {}: {} {:.3f}".format(name, self.metric_name, metric))
                if self.args.save_embedding:
                    h.remove()
                    assert len(embeddings) == len(loader), 'embedding size not consistent with data size'
                    data = {'feat': torch.cat(embeddings, dim=0).numpy(),
                     'label': torch.cat(labels, dim=0).numpy()}
                    np.save(os.path.join(self.ckpoint_folder, "{}_seed{}.npy".format(name, self.args.rand_seed)), data)

        target_metric_avg = 0
        target_count = 0
        for name, metric in metrics.items():
            if name not in self.args.source:
                target_count += 1
                target_metric_avg += metric

        if target_count > 0:
            target_metric_avg /= target_count

        logs = []
        logs.append("\t".join(metrics.keys()))

        numbers = []
        for name, metric in metrics.items():
            numbers.append("{:.3f}".format(metric))
        logs.append("\t".join(numbers))
        if target_count > 0:
            logs.append('Target Domain average performance: {:.3f}'.format(target_metric_avg))
        self.log('\n'.join(logs))


    def test_corrupted(self, net, testloaders, severity=1):
        """
        Testing on corrupted data
        """

        c_types = testloaders.keys()

        metrics = {name: [] for name in c_types}

        with tqdm(testloaders.items(), leave=True, desc='Severity {}: '.format(severity)) as d_iter:
            for name, loader in d_iter:
                metric, _ = self.infer(net, self.device, loader, name='{}-{}'.format(severity, name.split('_')[0]))
                metrics[name].append(metric)
                d_iter.set_postfix_str("Corruption {}: {} {:.3f}".format(name, self.metric_name, metric))

        target_metric_avg = 0
        target_count = 0
        for name, metric in metrics.items():
            a = np.array(metric)
            target_count += 1
            target_metric_avg += a

        logs = []
        logs.append("\t".join(metrics.keys()))

        if target_count > 0:
            target_metric_avg /= target_count

        numbers = []
        for name, metric in metrics.items():
            numbers.append("{:.3f}".format(metric[0]))
        logs.append("\t".join(numbers))
        if target_count > 0:
            logs.append('Corruption severity {} average performance: {:.3f}'.format(severity, target_metric_avg[0]))
        self.log('\n'.join(logs))

    def resume_model(self, net, test_latest=False, test_target=False):
        # Load checkpoint for testing
        assert os.path.isdir(self.ckpoint_folder), 'Error: no checkpoint directory {} found!'.format(self.ckpoint_folder)
        if test_latest:
            ckpoint_file = self.last_ckpoint
        elif test_target:
            ckpoint_file = self.best_target_ckpoint
        else:
            ckpoint_file = self.best_ckpoint

        print('==> Resuming from checkpoint {}'.format(self.ckpoint_folder))
        checkpoint = torch.load(ckpoint_file)
        if 'rand_module' in checkpoint and self.rand_module is not None:
            self.rand_module.load_state_dict(checkpoint['rand_module'], strict=False)

        net.load_state_dict(checkpoint['net'], strict=False)
        best_metric = checkpoint[self.metric_name]
        best_epoch = checkpoint['epoch']
        print("{} {} at {} epoch".format(self.metric_name, best_metric, best_epoch))

        # log epoch of tested checkpoint
        if test_latest:
            self.log("\nThe last model at {} epoch".format(best_epoch))
        else:
            self.log("\nThe best model at {} epoch".format(best_epoch))


    def log(self, string):
        print(string)
        f = open(self.log_path, mode='a')
        f.write(str(string))
        f.write('\n')
        f.close()

    @staticmethod
    def print_now():
        print(datetime.now())
