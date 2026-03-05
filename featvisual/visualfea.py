from __future__ import division,print_function,unicode_literals
import os,sys,argparse,cv2
import numpy as np
import torch
sys.path.append('../')
from configms.modelna.segmodels import *
from configms.database.datahelpers import *
from utils.tvutils import *
from configms.torchcam.grad_cam import GradCAM
from configms.torchcam.hirescam import HiResCAM
from configms.torchcam.score_cam import ScoreCAM
from configms.torchcam.grad_cam_plusplus import GradCAMPlusPlus
from configms.torchcam.ablation_cam import AblationCAM
from configms.torchcam.xgrad_cam import XGradCAM
from configms.torchcam.eigen_cam import EigenCAM
from configms.torchcam.eigen_grad_cam import EigenGradCAM
from configms.torchcam.layer_cam import LayerCAM
from configms.torchcam.fullgrad_cam import FullGrad
from configms.torchcam.grad_cam_elementwise import GradCAMElementWise
from configms.torchcam.guided_backprop import GuidedBackpropReLUModel
from configms.torchcam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from configms.torchcam.utils.model_targets import ClassifierOutputTarget


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser('the test of medical segments')
    parser.add_argument('-i',dest='inputs',type=str,default='/media/tyy/ff587189-3630-4a5f-9322-4290b95c5cf3/CODE/medicalsegs/datasets/for_test/attention_test/images')
    # parser.add_argument('-f',dest='folders',nargs='+',type=str,default=['images','mask'])
    parser.add_argument('-na',dest='modelna',type=str,default='gcnpps')
    parser.add_argument('-md',dest='method',type=str,default='gradcam')
    parser.add_argument('-n',dest='numclasses',type=int,default=6)
    parser.add_argument('-o',dest='outputs',type=str,default='/media/tyy/ff587189-3630-4a5f-9322-4290b95c5cf3/CODE/borehole_Ablation_Study/Engsegs_scse/datasets/outputs/Task03')
    parser.add_argument('-sv',dest='saveimgs',type=str,default='/media/tyy/ff587189-3630-4a5f-9322-4290b95c5cf3/CODE/medicalsegs/datasets/for_test/attention_test/visual')
    parser.add_argument('-u',dest='use_cuda', action='store_true', default=True)
    parser.add_argument('-es', dest='eigen_smooth', action='store_true')
    parser.add_argument('-as',dest='aug_smooth', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    makedirs_func(os.path.join(args.saveimgs, args.modelna))

    methods = {"gradcam": GradCAM,
         "hirescam":HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    swp = os.path.join(args.outputs,'{}_swp'.format(args.modelna),'best_models.pth')
    model = SegModels(args,args.modelna)


    if torch.cuda.device_count() > 0:
        model = model.cuda()
        model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    else:
        model.cuda().to(device)

    model.load_state_dict(torch.load(swp))
    # print(model)
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model.eval()

    modellayers = 'blocks'
    target_layers = [model.encoder.layer1, model.encoder.layer2, model.encoder.layer3]
    target_layers = [model.encoder.layer1, model.encoder.layer2, model.encoder.layer3, model.encoder.layer4]
    target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_0_1, model.decoder.blocks.x_0_2, model.decoder.blocks.x_0_3, model.decoder.blocks.x_0_4]
    target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_1_1, model.decoder.blocks.x_1_2, model.decoder.blocks.x_1_3, model.decoder.blocks.x_0_4]
    target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_1_1, model.decoder.blocks.x_2_2, model.decoder.blocks.x_3_3, model.decoder.blocks.x_0_4]
    # target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_1_1, model.decoder.blocks.x_1_2, model.decoder.blocks.x_1_3, model.decoder.blocks.x_2_2, model.decoder.blocks.x_3_3, model.decoder.blocks.x_0_4]
    
    # target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_0_1, model.decoder.blocks.x_1_1, model.decoder.blocks.x_1_2, model.decoder.blocks.x_1_3, model.decoder.blocks.x_2_2, model.decoder.blocks.x_3_3, model.decoder.blocks.x_0_4.attention2]
    # target_layers = [model.decoder.blocks.x_0_0, model.decoder.blocks.x_0_1, model.decoder.blocks.x_1_1, model.decoder.blocks.x_1_2, model.decoder.blocks.x_1_3, model.decoder.blocks.x_2_2, model.decoder.blocks.x_2_3, model.decoder.blocks.x_3_3, model.decoder.blocks.x_0_4.attention2]
    # target_layers = [model.decoder]

    for names in tqdm(os.listdir(args.inputs)):
        rgb_img = cv2.imread(os.path.join(args.inputs,names))
        # rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


        # targets = None
        targets = [ClassifierOutputTarget(args.numclasses)]
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # targets = [ClassifierOutputTarget(args.numclasses)]

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=args.numclasses)

        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        cv2.imwrite(os.path.join(args.saveimgs, args.modelna,'{}_{}_{}_{}_cam.jpg'.format(names[:-4], args.modelna, args.method,modellayers)), cam_image)
        # cv2.imwrite(os.path.join(args.saveimgs,'{}_{}_gb.jpg'.format(names[:-4],args.method)), gb)
        # cv2.imwrite(os.path.join(args.saveimgs,'{}_{}_cam_gb.jpg'.format(names[:-4],args.method)), cam_gb)





if __name__ == '__main__':
    main()

