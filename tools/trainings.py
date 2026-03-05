#!/usr/bin/env python
from __future__ import division,print_function,unicode_literals
import os,sys,argparse,random,shutil,time
sys.path.append('../')
from configms.modelna.segmodels import *
from utils.dice import *
from utils.tvutils import *
from utils.metrics import *
# from tensorboardX import SummaryWriter
# from torchstat import stat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Multclass of images segmentation for borehole images')
    parser.add_argument('-i',dest='inputs',type=str,default='../datasets/inputs/')
    parser.add_argument('-f',nargs='+',dest='folders',type=str,default=['train','val','test'])
    parser.add_argument('-fi',nargs="+",dest='files',type=str,default=['images','mask'])
    parser.add_argument('-u',dest='use_cuda', action='store_true', default=True)
    parser.add_argument('-b',dest='batch_size',type=int,default=2)
    parser.add_argument('-n',dest='numclasses',type=int,default=6)
    parser.add_argument('-e',dest='epochs',type=int,default=100)
    parser.add_argument('-na',dest='modelna',type=str,default='gcnpps')
    parser.add_argument('-o',dest='outputs',type=str,default='../')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    saves = os.path.join(args.outputs,"{}_swp".format(args.modelna))
    makedirs_func(saves)

    train_dataset = Medical_Dataset(os.path.join(args.inputs,args.folders[0]),
        sub_dir = 'images' ,img_suffix = '.png',
        ann_dir = os.path.join(args.inputs,args.folders[0],args.files[1]),
        size = 512,debug=False)
    val_dataset = Medical_Dataset(os.path.join(args.inputs,args.folders[1]),
        sub_dir = 'images', img_suffix = '.png',
        ann_dir = os.path.join(args.inputs,args.folders[1],args.files[1]),
        size = 512,debug=False, test_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,drop_last=True)


    model = SegModels(args,args.modelna)
    if args.use_cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    else:
        model.cuda().to(device)
    
    ce_weight = torch.tensor(weightsAc(train_loader,args.numclasses),dtype=torch.float32).to(device)
    # criterion = DiceLoss(weight=ce_weight, ignore_index=-1)
    # criterion = Multi_MultiHeadCELoss(weight=ce_weight, loss2=True, loss2_weight=1.0)
    # criterion_1 = CrossEntropyLoss(weight=ce_weight)
    criterion = DiceLoss(mode='multiclass')
    # criterion = criterion_2+0.3*criterion_1
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4, weight_decay=1e-3)



    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 60, 70, 80, 90, 96], gamma=0.5)

    TrainEpoch(args,model, criterion, optimizer, scheduler,train_loader,valid_loader,saves)




if __name__ == '__main__':
    main()

