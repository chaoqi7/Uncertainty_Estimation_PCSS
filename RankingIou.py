import argparse
import glob
import os
import numpy as np

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--in_data_dir', type=str,
                        default='/home/qc/uncert/PointNetand2/data/stanford_indoor3d_sy2/',
                        help='in pred dir path')
    parser.add_argument('--in_pred_dir', type=str, default='/home/qc/uncert/Uncertainty Estimation(GitHub)/log/sem_seg/2021-03-09_05-18/output_val/', help='in pred dir path')
    parser.add_argument('--sample_times', type=int,
                        default= 10,
                        help='in pred dir path')
    parser.add_argument('--vox_nums', type=int,
                        default= 5,
                        help='in pred dir path')
    parser.add_argument('--in_pred_dir_vox', type=str,
                        default='/home/qc/uncert/Uncertainty Estimation(GitHub)/log/sem_seg/2021-03-09_05-18/output_val/vox',
                        help='in pred dir vox path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data_names = glob.glob(os.path.join(args.in_data_dir, '*'))
    label_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir,'label'), '*'))
    pred_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'pred'), '*'))
    unct_b_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_b'), '*'))
    unct_e_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir,'unct_e'), '*'))
    unct_r_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_r'), '*'))
    unct_v_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_v'), '*'))



    datas = np.array([], dtype='uint8')
    labels= np.array([], dtype='uint8')

    preds = np.array([], dtype='uint8')
    unct_bs = np.array([], dtype='uint8')
    unct_es= np.array([], dtype='uint8')
    unct_rs = np.array([], dtype='uint8')
    unct_vs = np.array([], dtype='uint8')

    equrate2 = np.array([], dtype='uint8')
    voxconpoint2 = np.array([], dtype='uint8')
    unct_bs2 = np.array([], dtype='uint8')
    unct_es2 = np.array([], dtype='uint8')
    unct_rs2 = np.array([], dtype='uint8')
    unct_vs2 = np.array([], dtype='uint8')

    for index, pred_name in enumerate(pred_names):
        if 'NSA_' in pred_name:
            pred = np.load(pred_name)
            choice = np.random.choice(np.shape(pred)[0], int(np.shape(pred)[0] / args.sample_times), replace=False)
            preds = np.append(preds, pred[choice])
            unct_b = np.load(unct_b_names[index])
            unct_bs = np.append(unct_bs, unct_b[choice])
            unct_e = np.load(unct_e_names[index])
            unct_es = np.append(unct_es, unct_e[choice])
            unct_r = np.load(unct_r_names[index])
            unct_rs = np.append(unct_rs, unct_r[choice])
            unct_v = np.load(unct_v_names[index])
            unct_vs = np.append(unct_vs, unct_v[choice])
            ### original data reading ###
            tempt = pred_name.split('NSA_')
            data = np.load(args.in_data_dir+tempt[1])
            data[:, 0] = data[:, 0] - np.min(data[:, 0])
            data[:, 1] = data[:, 1] - np.min(data[:, 1])
            data[:, 2] = data[:, 2] - np.min(data[:, 2])
            if datas.shape[0]==0:
                datas = data[choice,0:3]
            else:
                datas = np.concatenate((datas, data[choice,0:3]),axis=0)
            label = np.load(label_names[index])
            labels = np.append(labels, label[choice])

    x_min = np.min(datas[:,0])
    x_max = np.max(datas[:,0])
    y_min = np.min(datas[:,1])
    y_max = np.max(datas[:,1])
    z_min = np.min(datas[:,2])
    z_max = np.max(datas[:,2])
    x_step=(x_max-x_min)/args.vox_nums
    y_step = (y_max - y_min) / args.vox_nums
    z_step = (z_max - z_min) / args.vox_nums
    for x in range(0, args.vox_nums):
        for y in range(0, args.vox_nums):
            for z in range(0, args.vox_nums):
                lx,ly,lz=x_min+x*x_step,y_min+y*y_step,z_min+z*z_step
                hx,hy,hz=lx+x_step, ly+y_step,lz+z_step
                cors = np.argwhere(((datas[:, 0] >= lx) &(datas[:, 0] < hx) &(datas[:, 1] >= ly) &(datas[:, 1] < hy) &(datas[:, 2] >= lz) &(datas[:, 2] < hz))==True)
                if(len(cors)>0):
                    label_tempt = labels[cors]
                    pred_mydpwithsimilar_tempt = preds[cors]

                    equ_mydpwithsimilar_nums=np.shape(np.argwhere((label_tempt==pred_mydpwithsimilar_tempt)==True))[0]
                    equ_mydpwithsimilar_rate=equ_mydpwithsimilar_nums/np.shape(label_tempt)[0]


                    unct_b_mydpwithsimilar_tempt = np.mean(unct_bs[cors][np.where(~np.isnan(unct_bs[cors]))])
                    unct_e_mydpwithsimilar_tempt = np.mean(unct_es[cors][np.where(~np.isnan(unct_es[cors]))])
                    unct_r_mydpwithsimilar_tempt = np.mean(unct_rs[cors][np.where(~np.isnan(unct_rs[cors]))])
                    unct_v_mydpwithsimilar_tempt = np.mean(unct_vs[cors][np.where(~np.isnan(unct_vs[cors]))])

                    equrate2 = np.append(equrate2, equ_mydpwithsimilar_rate)
                    voxconpoint2 = np.append(voxconpoint2, len(cors))
                    unct_bs2 = np.append(unct_bs2, unct_b_mydpwithsimilar_tempt)
                    unct_es2 = np.append(unct_es2 , unct_e_mydpwithsimilar_tempt)
                    unct_rs2 = np.append(unct_rs2, unct_r_mydpwithsimilar_tempt)
                    unct_vs2 = np.append(unct_vs2, unct_v_mydpwithsimilar_tempt)


    np.save(os.path.join(args.in_pred_dir_vox, 'vox_equrate'+ str(args.vox_nums)), equrate2)
    np.save(os.path.join(args.in_pred_dir_vox, 'vox_unct_b'+ str(args.vox_nums)), unct_bs2)
    np.save(os.path.join(args.in_pred_dir_vox, 'vox_unct_e'+ str(args.vox_nums)), unct_es2)
    np.save(os.path.join(args.in_pred_dir_vox, 'vox_unct_r'+ str(args.vox_nums)), unct_rs2)
    np.save(os.path.join(args.in_pred_dir_vox, 'vox_unct_v'+ str(args.vox_nums)), unct_vs2)
