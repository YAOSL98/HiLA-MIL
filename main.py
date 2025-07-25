import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from torch.nn.functional import one_hot
from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils import *
import torch.nn.functional as F
def main(args):
    # set seed
    seed_torch(args.seed)

    # --->get dataset
    if args.datasets.lower() == 'camelyon16':
        label_path=os.path.join(args.dataset_root,'all.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    elif args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]

    if not args.no_log:
        print('Dataset: ' + args.datasets)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        if len(ckp['ckc_metric']) == 6:
            acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 7:
            acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
        else:
            acs, pre, rec,fs,auc = ckp['ckc_metric']

    for k in range(args.fold_start, args.cv_fold):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

    if args.always_test:
        if args.wandb:
            wandb.log({
                "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                "cross_val/te_auc_std":np.std(np.array(te_auc)),
                "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                "cross_val/te_f1_std":np.std(np.array(te_fs)),
            })

    if args.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/recall_std":np.std(np.array(rec)),
        })
    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
    # --->initiation
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

    # --->load data
    if args.datasets.lower() == 'camelyon16':

        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    elif args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mm_sche = None
    if not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init

    # --->bulid networks
    if args.model == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)

        model = mhim.MHIM(**model_params).to(device)
            
    elif args.model == 'pure':
        model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)

    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        if not args.no_log:
            print(info)

    # teacher model
    if args.model == 'mhim':
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            if not args.no_log:
                print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                info = model_tea.load_state_dict(pre_dict,strict=False)
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        if args.tea_type == 'same':
            model_tea = model
    else:
        model_tea = None

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = 0, 0, 0, 0,0,0
    opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        if len(ckp['te_best_metric']) > 1:
            opt_te_fs = ckp['te_best_metric'][1]
        opt_te_tea_auc,opt_te_tea_fs = ckp['te_best_metric'][2:4]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False

    train_time_meter = AverageMeter()

    for epoch in range(epoch_start, args.num_epoch):
        train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        train_time_meter.update(end-start)
        stop,accuracy, auc_value, precision, recall, fscore, test_loss = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)

        if model_tea is not None:
            _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            if args.wandb:
                rowd = OrderedDict([
                    ("val_acc_tea",accuracy_tea),
                    ("val_precision_tea",precision_tea),
                    ("val_recall_tea",recall_tea),
                    ("val_fscore_tea",fscore_tea),
                    ("val_auc_tea",auc_value_tea),
                    ("val_loss_tea",test_loss_tea),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if auc_value_tea > opt_tea_auc:
                opt_tea_auc = auc_value_tea
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_tea_auc",opt_tea_auc)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

        if args.always_test:

            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
            
            if args.wandb:
                rowd = OrderedDict([
                    ("te_acc",_te_accuracy),
                    ("te_precision",_te_precision),
                    ("te_recall",_te_recall),
                    ("te_fscore",_te_fscore),
                    ("te_auc",_te_auc_value),
                    ("te_loss",_te_test_loss_log),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_te_auc",opt_te_auc),
                        ("best_te_f1",_te_fscore)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)
            
            if model_tea is not None:
                _te_tea_accuracy, _te_tea_auc_value, _te_tea_precision, _te_tea_recall, _te_tea_fscore,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
                if args.wandb:
                    rowd = OrderedDict([
                        ("te_tea_acc",_te_tea_accuracy),
                        ("te_tea_precision",_te_tea_precision),
                        ("te_tea_recall",_te_tea_recall),
                        ("te_tea_fscore",_te_tea_fscore),
                        ("te_tea_auc",_te_tea_auc_value),
                        ("te_tea_loss",_te_tea_test_loss_log),
                    ])

                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

                if _te_tea_auc_value > opt_te_tea_auc:
                    opt_te_tea_auc = _te_tea_auc_value
                    opt_te_tea_fs = _te_tea_fscore
                    if args.wandb:
                        rowd = OrderedDict([
                            ("best_te_tea_auc",opt_te_tea_auc),
                            ("best_te_tea_f1",_te_fscore)
                        ])
                        rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd)
        if not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        if args.wandb:
            rowd = OrderedDict([
                ("val_acc",accuracy),
                ("val_precision",precision),
                ("val_recall",recall),
                ("val_fscore",fscore),
                ("val_auc",auc_value),
                ("val_loss",test_loss),
                ("epoch",epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)

        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        if epoch > 0:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_epoch_{epoch}_model_best_auc.pt'.format(fold=k, epoch=epoch)))

        if args.wandb:
            rowd = OrderedDict([
                ("val_best_acc",optimal_ac),
                ("val_best_precesion",opt_pre),
                ("val_best_recall",opt_re),
                ("val_best_fscore",opt_fs),
                ("val_best_auc",opt_auc),
                ("val_best_epoch",opt_epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
            'te_best_metric': [opt_te_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs],
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break
    
    # test
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        info = model.load_state_dict(best_std['model'])
        print(info)
        if model_tea is not None and best_std['teacher'] is not None:
            info = model_tea.load_state_dict(best_std['teacher'])
            print(info)

    accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
    
    if args.wandb:
        wandb.log({
            "test_acc":accuracy,
            "test_precesion":precision,
            "test_recall":recall,
            "test_fscore":fscore,
            "test_auc":auc_value,
            "test_loss":test_loss_log,
        })
    if not args.no_log:
        print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)

    if args.always_test:
        te_auc.append(opt_te_auc)
        te_fs.append(opt_te_fs)
        
    return [acs,pre,rec,fs,auc,te_auc,te_fs]

def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    if model_tea is not None:
        model_tea.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if isinstance(data[0],(list,tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag=data[0]
            batch_size=data[0][0].size(0)
        else:
            bag=data[0].to(device)  # b*n*1024
            batch_size=bag.size(0)
            
        label=data[1].to(device)
        
        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'mhim':
                if model_tea is not None:
                    cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True)
                else:
                    attn,cls_tea = None,None

                cls_tea = None if args.cl_alpha == 0. else cls_tea

                train_logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)

            elif args.model == 'pure':
                train_logits, cls_loss,patch_num,keep_num = model.pure(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                train_logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            else:
                train_logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.

            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size,-1),label)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

        train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'mhim':
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
                if model_tea is not None:
                    if args.tea_type == 'same':
                        pass
                    else:
                        ema_update(model,model_tea,mm)
            else:
                mm = 0.

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    epoch_time = end-start
    print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    start_time = time.time()
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []
    bag_prob = []
    bag_hat = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    # bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    Y_hat = torch.argmax(test_logits, dim=1)
                    Y_prob = F.softmax(test_logits, dim=1)
                    bag_prob.append(Y_prob.cpu().squeeze().numpy())
                    bag_hat.append(Y_hat.cpu().squeeze())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)
    
    # save the log file
    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_hat, bag_prob) 
    
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    end_time = time.time()  # 记录结束时间
    val_duration = end_time - start_time  # 计算时间差

    print(f"Validation loop took {val_duration:.2f} seconds")
    return stop,accuracy, auc_value, precision, recall, fscore,loss_cls_meter.avg

def test(args,model,loader,device,criterion,model_tea=None):
    start_time = time.time()
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []
    bag_prob = []
    bag_hat = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())

                    # bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    Y_hat = torch.argmax(test_logits, dim=1)
                    Y_prob = F.softmax(test_logits, dim=1)
                    bag_prob.append(Y_prob.cpu().squeeze().numpy())
                    bag_hat.append(Y_hat.cpu().squeeze())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_hat, bag_prob) 
    test_loss_log = test_loss_log/len(loader)

    end_time = time.time()  # 记录结束时间
    val_duration = end_time - start_time  # 计算时间差

    print(f"Validation loop took {val_duration:.2f} seconds")

    return accuracy, auc_value, precision, recall, fscore,test_loss_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')

    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='mhim', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    
    # Siamese framework
    parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')

    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if args.model == 'pure':
        args.cl_alpha=0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    if args.datasets == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True

    if args.wandb:
        if args.auto_resume:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
        else:
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path))
        
    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    main(args=args)
