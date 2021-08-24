import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import ceil, log10
from dataset import get_test_Set
from utils import *
from torchvision.utils import save_image
from model import SBMENet, ABMRNet, SynthesisNet
from torch.backends import cudnn
import argparse

cudnn.benchmark = True

def config_ME(args):
    if args.Dataset in ['ucf101','vimeo']:
        args.divisor = 64.
        args.D_factor = 1.
        args.margin = 0 
    elif args.Dataset in ['SNU-FILM-all','Xiph_HD']:
        args.divisor = 128.
        args.D_factor = 0.5
        args.margin = 0
    elif args.Dataset in ['X4K1000FPS']:
        args.divisor = 256.
        args.D_factor = 0.25
        args.margin = 1
    return args

def test(args):
    avg_psnr = 0
    MSE = nn.MSELoss().cuda()
    
    SBMNet = SBMENet()
    ABMNet = ABMRNet()
    SynNet = SynthesisNet(args)
        
    SBMNet.load_state_dict(torch.load(args.SBMNet_ckpt, map_location='cpu'))
    ABMNet.load_state_dict(torch.load(args.ABMNet_ckpt, map_location='cpu'))
    SynNet.load_state_dict(torch.load(args.SynNet_ckpt, map_location='cpu'))
    
    for param in SBMNet.parameters():
        param.requires_grad = False # Freeze the SBM Net
    for param in ABMNet.parameters():
        param.requires_grad = False # Freeze the ABM Net
    for param in SynNet.parameters():
        param.requires_grad = False # Freeze the Syn Net
        
    SBMNet.cuda()
    ABMNet.cuda()
    SynNet.cuda()

    test_set = get_test_Set(args)
    validation_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)

    with torch.no_grad():
        for iteration, batch in enumerate(validation_data_loader, 1):
            input = batch[0].cuda()

            frame1 = input[:, :3, :, :]   
            frame3 = input[:, 3:6, :, :]  
            
            W = frame1.shape[3]
            H = frame1.shape[2]
            
            H_ = int(ceil(H / args.divisor) * args.divisor * args.D_factor)        
            W_ = int(ceil(W / args.divisor) * args.divisor * args.D_factor)        
            
            frame1_ = F.interpolate(frame1, (H_, W_), mode='bilinear')
            frame3_ = F.interpolate(frame3, (H_, W_), mode='bilinear')
            
            SBM = SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
            SBM_ = F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

            frame2_1, Mask2_1 = warp(frame1_, SBM_*(-1),  return_mask=True)
            frame2_3, Mask2_3 = warp(frame3_, SBM_,       return_mask=True)

            frame2_Anchor_ = (frame2_1 + frame2_3) / 2
            frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1-Mask2_1) + frame2_1 * (1-Mask2_3))

            Z = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
            Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)
            
            ABM_bw, _ = ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM*(-1), Z_.exp())
            ABM_fw, _ = ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

            SBM_     = F.interpolate(SBM, (H, W), mode='bilinear')    * 20.0
            ABM_fw   = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
            ABM_bw   = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

            SBM_[:, 0, :, :] *= W / float(W_)
            SBM_[:, 1, :, :] *= H / float(H_)
            ABM_fw[:, 0, :, :] *= W / float(W_)
            ABM_fw[:, 1, :, :] *= H / float(H_)
            ABM_bw[:, 0, :, :] *= W / float(W_)
            ABM_bw[:, 1, :, :] *= H / float(H_)

            s_divisor = 8.
            H_ = int(ceil(H / s_divisor) * s_divisor)
            W_ = int(ceil(W / s_divisor) * s_divisor)
            
            Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)
            
            Syn_inputs = F.interpolate(Syn_inputs, (H_,W_), mode='bilinear')
            Syn_inputs[:, 6, :, :] *= float(W_) / W
            Syn_inputs[:, 7, :, :] *= float(H_) / H
            Syn_inputs[:, 8, :, :] *= float(W_) / W
            Syn_inputs[:, 9, :, :] *= float(H_) / H
            Syn_inputs[:, 10, :, :] *= float(W_) / W
            Syn_inputs[:, 11, :, :] *= float(H_) / H 

            output = SynNet(Syn_inputs)
    
            I2 = F.interpolate(output, (H,W), mode='bicubic')
            
            ## This is not accurate PSNR.
            ## If you want to evaluate accurate performance,
            ## you should save images and then compare these with ground truth.
            mse = MSE(I2, batch[1].cuda())
            psnr = 10 * log10(1/mse.item())
            avg_psnr += psnr

            if args.is_save:
                if args.Dataset in ['ucf101']:
                    save_image(I2 , batch[-1][0].replace('.png','_%s.png'%args.name))
                elif args.Dataset in ['vimeo']:
                    save_image(I2 , batch[-1][0].replace('target',args.name).replace('.png','_%s.png'%args.name))
                elif args.Dataset in ['SNU-FILM-all', 'Xiph_HD']:
                    save_image(I2 , batch[-1][0])

            if iteration % 100 == 0:
                if args.is_save:
                    print('[%s:%s](\033[1;32;1m%d\033[0m/%d) is finished and saved' % (args.name, args.Dataset, iteration, len(validation_data_loader)))
                else:
                    print('[%s:%s](\033[1;32;1m%d\033[0m/%d) is finished' % (args.name, args.Dataset, iteration, len(validation_data_loader)))
    
    print('[%s:%s] avg. PSNR: %6f' %(args.name, args.Dataset, avg_psnr/len(validation_data_loader)))

    with open('%s_test_log.txt'%args.name,'a') as txt:
        txt.write('[%s:%s:%s:%s] avg.PSNR: %6f\n'%(args.Dataset,args.SBMNet_ckpt, args.ABMNet_ckpt, args.SynNet_ckpt, avg_psnr/len(validation_data_loader)))

def test_X4K1000FPS(args):
    mul_list = [(0,16,32), (0,8,16), (16,24,32), (0,4,8), (8,12,16), (16,20,24), (24,28,32)]

    SBMNet = SBMENet()
    ABMNet = ABMRNet()
    SynNet = SynthesisNet(args)
        
    SBMNet.load_state_dict(torch.load(args.SBMNet_ckpt, map_location='cpu'))
    ABMNet.load_state_dict(torch.load(args.ABMNet_ckpt, map_location='cpu'))
    SynNet.load_state_dict(torch.load(args.SynNet_ckpt, map_location='cpu'))
    
    for param in SBMNet.parameters():
        param.requires_grad = False # Freeze the SBM Net
    for param in ABMNet.parameters():
        param.requires_grad = False # Freeze the ABM Net
    for param in SynNet.parameters():
        param.requires_grad = False # Freeze the Syn Net
        
    SBMNet.cuda()
    ABMNet.cuda()
    SynNet.cuda()

    test_set = get_test_Set(args)
    validation_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)

    with torch.no_grad():
        for iteration, batch in enumerate(validation_data_loader, 1):
            seq_dict = dict()
            
            seq_dict[0], seq_dict[32] = batch[0].split([3,3], dim=1)

            for mul in mul_list:         
                frame1_full = seq_dict[mul[0]].cuda()
                frame3_full = seq_dict[mul[2]].cuda()

                W_full = frame1_full.shape[3]
                H_full = frame1_full.shape[2]
                
                W_Half = W_full // 2
                H_Half = H_full // 2

                H_margin = int((ceil(H_Half / args.divisor)+args.margin) * args.divisor) - H_Half
                W_margin = int((ceil(W_Half / args.divisor)+args.margin) * args.divisor) - W_Half

                I2_list = []

                for direction in ['nw','ne','sw','se']:
                    
                    if direction == 'nw':
                        frame1 = frame1_full[:,:,:H_Half+H_margin,:W_Half+W_margin]
                        frame3 = frame3_full[:,:,:H_Half+H_margin,:W_Half+W_margin]
                    elif direction == 'ne':
                        frame1 = frame1_full[:,:,:H_Half+H_margin,W_Half-W_margin:]
                        frame3 = frame3_full[:,:,:H_Half+H_margin,W_Half-W_margin:]
                    elif direction == 'sw':
                        frame1 = frame1_full[:,:,H_Half-H_margin:,:W_Half+W_margin]
                        frame3 = frame3_full[:,:,H_Half-H_margin:,:W_Half+W_margin]
                    elif direction == 'se':
                        frame1 = frame1_full[:,:,H_Half-H_margin:,W_Half-W_margin:]
                        frame3 = frame3_full[:,:,H_Half-H_margin:,W_Half-W_margin:]

                    W = frame1.shape[3]
                    H = frame1.shape[2]
                    
                    H_ = int(ceil(H / args.divisor) * args.divisor * args.D_factor)
                    W_ = int(ceil(W / args.divisor) * args.divisor * args.D_factor)

                    frame1_ = F.interpolate(frame1, (H_, W_), mode='bilinear')
                    frame3_ = F.interpolate(frame3, (H_, W_), mode='bilinear')
                    
                    SBM = SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
                    SBM_ = F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

                    frame2_1, Mask2_1 = warp(frame1_, SBM_*(-1),  return_mask=True)
                    frame2_3, Mask2_3 = warp(frame3_, SBM_,       return_mask=True)

                    frame2_Anchor_ = (frame2_1 + frame2_3) / 2
                    frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1-Mask2_1) + frame2_1 * (1-Mask2_3))

                    Z = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
                    Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)
                    
                    ABM_bw, _ = ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM*(-1), Z_.exp())
                    ABM_fw, _ = ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

                    SBM_     = F.interpolate(SBM, (H, W), mode='bilinear')    * 20.0
                    ABM_fw   = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
                    ABM_bw   = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

                    SBM_[:, 0, :, :] *= W / float(W_)
                    SBM_[:, 1, :, :] *= H / float(H_)
                    ABM_fw[:, 0, :, :] *= W / float(W_)
                    ABM_fw[:, 1, :, :] *= H / float(H_)
                    ABM_bw[:, 0, :, :] *= W / float(W_)
                    ABM_bw[:, 1, :, :] *= H / float(H_)

                    s_divisor = 8.
                    H_ = int(ceil(H / s_divisor) * s_divisor)
                    W_ = int(ceil(W / s_divisor) * s_divisor)
                    
                    Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)
                    
                    Syn_inputs = F.interpolate(Syn_inputs, (H_,W_), mode='bilinear')
                    Syn_inputs[:, 6, :, :] *= float(W_) / W
                    Syn_inputs[:, 7, :, :] *= float(H_) / H
                    Syn_inputs[:, 8, :, :] *= float(W_) / W
                    Syn_inputs[:, 9, :, :] *= float(H_) / H
                    Syn_inputs[:, 10, :, :] *= float(W_) / W
                    Syn_inputs[:, 11, :, :] *= float(H_) / H 

                    output = SynNet(Syn_inputs)
            
                    I2_list.append(F.interpolate(output, (H,W), mode='bicubic').cpu())
                                        
                
                I2_N = torch.cat((I2_list[0][:,:,:H_Half,:W_Half], I2_list[1][:,:,:H_Half,W_margin:]), dim=3)  # NW+NE
                I2_S = torch.cat((I2_list[2][:,:,H_margin:,:W_Half], I2_list[3][:,:,H_margin:,W_margin:]), dim=3)  # SW+SE
                I2 = torch.cat((I2_N, I2_S), dim=2) # N + S    

                if args.is_save:
                    save_image(I2, '%s/%04d.png'%(batch[-1][0],mul[1]))

                seq_dict[mul[1]] = I2

            print('[%s:X4K1000FPS](\033[1;32;1m%d\033[0m/%d) is finished' % (args.name, iteration, len(validation_data_loader)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ABME', help="name your experiment")
    parser.add_argument('--Dataset', type=str, required=True, choices=['ucf101','vimeo','SNU-FILM-all','Xiph_HD','X4K1000FPS'])
    parser.add_argument('--dataset_root', type=str, required=True) 
    parser.add_argument('--SBMNet_ckpt', type=str, default='Best/SBME_ckpt.pth')
    parser.add_argument('--ABMNet_ckpt', type=str, default='Best/ABMR_ckpt.pth')
    parser.add_argument('--SynNet_ckpt', type=str, default='Best/SynNet_ckpt.pth')
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--DDP', action='store_true')
    
    args = parser.parse_args()
    args = config_ME(args)

    if args.Dataset == 'X4K1000FPS':
        if args.is_save:
            test_X4K1000FPS(args)
        else:
            raise ValueError('You should parse the \'is_save\' argument for X4K1000FPS test.')
    else:
        test(args)