from __future__ import division,print_function,unicode_literals
import os,sys,time,torch,cv2
from skimage import io
sys.path.append('../')
from utils.metrics import *
from prettytable import PrettyTable
from torch.autograd import *
from tqdm import tqdm
from utils.pallette import *
from configms.database.datahelpers import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled = False

def writetxt(boxes, outputs):
    with open(outputs,'w',encoding='utf-8') as fts:
        fts.writelines(boxes)
    fts.close()

def TrainEpoch(args,model, criterion, optimizer, scheduler,train_loader,valid_loader,saves):
    all_iters = float(len(train_loader)*args.epochs)
    curr_epoch=0
    best_miou = 0
    best_f = 0

    while True:
        # torch.cuda.empty_cache()
        model.train()
        train_loss = AverageMeter()
        conf_mat = np.zeros((args.numclasses, args.numclasses)).astype(np.int64)

        curr_iters = curr_epoch * len(train_loader)
        for index,(images,masks,names) in enumerate(tqdm(train_loader)):
            running_iter = curr_iters + index + 1
            images, target = images.to(device).float(), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss_fn = criterion(outputs, target)
            train_loss.update(loss_fn.cpu().detach().numpy())
            loss_fn.backward()
            optimizer.step()


            _,preds = torch.max(outputs,dim=1)
            target = target.data.cpu().numpy().squeeze().astype(np.uint8)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += Confusion_matrix_res(target.flatten(),preds.flatten(),outputs.shape[1])

        acc, mean_IoU, kappa,fwiou,f1score = Evaluate_res(conf_mat)

        curr_epoch += 1
        scheduler.step()

        val_IoU, val_f = ValidEpoch(args,model, criterion, optimizer, scheduler,valid_loader,curr_epoch)

        if best_miou < val_IoU or best_f < val_f:
            best_miou = val_IoU
            best_f = val_f
            torch.save(model.state_dict(), os.path.join(saves,'best_models.pth'))
            print(" the model is saved.")
        else:
            continue

        if curr_epoch >= args.epochs:
            return


def ValidEpoch(args,model, criterion, optimizer, scheduler,valid_loader,curr_epoch):
    # torch.cuda.empty_cache()
    model.eval()
    valid_loss = AverageMeter()
    conf_mat = np.zeros((args.numclasses, args.numclasses)).astype(np.int64)

    for index,(images,masks,names) in enumerate(tqdm(valid_loader)):
        images, target = images.to(device).float(), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss_fn = criterion(outputs, target)
        valid_loss.update(loss_fn.cpu().detach().numpy())
        loss_fn.backward()
        optimizer.step()

        _,preds = torch.max(outputs,dim=1)
        target = target.data.cpu().numpy().squeeze().astype(np.uint8)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        conf_mat += Confusion_matrix_res(target.flatten(),preds.flatten(),outputs.shape[1])

    acc, mean_IoU, kappa,fwiou,f1score = Evaluate_res(conf_mat)
    print('curr_epoch={}, acc={}, mean_IoU={}, kappa={},fwiou={},f1score={}'.format(curr_epoch,acc, mean_IoU, kappa,fwiou,f1score))
    return mean_IoU, f1score


def TestdEpoch(args,model,test_loader):
    infor_pr = os.path.join(args.results, args.modelna,"infors/pred")
    infor_gt = os.path.join(args.results, args.modelna,"infors/gt")
    makedirs_func(infor_pr)
    makedirs_func(infor_gt)

    # torch.cuda.empty_cache()

    # with torch.no_grad():
    model.eval()
    cms = np.zeros((args.numclasses,args.numclasses)).astype(np.int64)
    for index,(images,target, names) in enumerate(tqdm(test_loader)):
        images, target = images.to(device).float(), target.to(device).long()
        outputs = model(images)
        _,preds = torch.max(outputs,dim=1)
        target = target.data.cpu().numpy().squeeze().astype(np.uint8)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

        vis_pred = Index2Color(preds)
        vis_gt = Index2Color(target)
        io.imsave(os.path.join(infor_pr,names[0]),vis_pred)
        io.imsave(os.path.join(infor_gt,names[0]),vis_gt)

        cms += Confusion_matrix_res(target.flatten(),preds.flatten(),outputs.shape[1])


    acc, mean_IoU, kappa,fwiou,f1score = Evaluate_res(cms)
    acc, mean_IoU, dice, sen, spe = Evaluates(cms)
    print('dice={}, mean_IoU={}, sen={}, spe={}, acc={}'.format(dice, mean_IoU, sen, spe, acc))
    boxes = str(round(dice*100, 2))+' '+str(round(mean_IoU*100, 2))+' '+str(round(sen*100, 2)) + ' '+str(round(spe*100, 2)) +' '+str(round(acc*100, 2))
    writetxt(boxes, os.path.join(args.results, args.modelna,'infors.txt'))


