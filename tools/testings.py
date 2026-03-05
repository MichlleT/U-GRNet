from __future__ import division,unicode_literals,print_function
import os,sys,argparse,time
sys.path.append('../')
from configms.modelna.segmodels import *
from utils.tvutils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser('the test of segments')
    parser.add_argument('-i',dest='inputs',type=str,default='../datasets/inputs/samples/test')
    parser.add_argument('-f',dest='folders',nargs='+',type=str,default=['images','mask'])
    parser.add_argument('-na',dest='modelna',type=str,default='gcnpps')
    parser.add_argument('-n',dest='numclasses',type=int,default=6)
    parser.add_argument('-o',dest='outputs',type=str,default='../checkpoint')
    parser.add_argument('-r',dest='results',type=str,default='../datasets/outputs/results')
    parser.add_argument('-b',dest='batch_size',type=int,default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    makedirs_func(args.results)

    swp = os.path.join(args.outputs,'{}_swp'.format(args.modelna),'best_models.pth')
    test_dataset = Medical_Dataset(args.inputs,
        sub_dir = 'images', img_suffix = '.png',
        ann_dir = os.path.join(args.inputs,args.folders[1]),
        size = 512,debug=False, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = GUnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=args.numclasses,  decoder_attention_type='scse') # decoder_attention_type='scse'

    if torch.cuda.device_count() > 0:
        model = model.cuda()
        model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    else:
        model.cuda().to(device)

    model.load_state_dict(torch.load(swp))

    TestdEpoch(args,model,test_loader)


if __name__ == '__main__':
    main()

