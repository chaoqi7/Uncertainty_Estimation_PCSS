import argparse
import glob
import os
import numpy as np

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--in_pred_dir', type=str, default='', help='in pred dir path')
    parser.add_argument('--in_pred_dir_summar', type=str,default='',help='in pred dir summar path')
    parser.add_argument('--sample_times', type=int, default= 10, help='sample_times')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    label_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir,'label'), '*'))
    pred_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'pred'), '*'))
    unct_b_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_b'), '*'))
    unct_e_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir,'unct_e'), '*'))
    unct_r_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_r'), '*'))
    unct_v_names = glob.glob(os.path.join(os.path.join(args.in_pred_dir, 'unct_v'), '*'))

    label_names= np.array([], dtype='uint8')
    pred_names = np.array([], dtype='uint8')
    unct_b_names = np.array([], dtype='uint8')
    unct_e_names= np.array([], dtype='uint8')
    unct_r_names = np.array([], dtype='uint8')
    unct_v_names = np.array([], dtype='uint8')

    for index, pred_name in enumerate(pred_names):
        if 'NSA_' in pred_name:
            pred = np.load(pred_name)
            choice = np.random.choice(np.shape(pred)[0], int(np.shape(pred)[0]/args.sample_times), replace=False)
            mcdp_pred_names = np.append(mcdp_pred_names, pred[choice])
            label = np.load(label_names[index])
            mcdp_label_names = np.append(mcdp_label_names, label[choice])
            unct_b = np.load(unct_b_names[index])
            mcdp_unct_b_names = np.append(mcdp_unct_b_names, unct_b[choice])
            unct_e = np.load(unct_e_names[index])
            mcdp_unct_e_names = np.append(mcdp_unct_e_names, unct_e[choice])
            unct_r = np.load(unct_r_names[index])
            mcdp_unct_r_names = np.append(mcdp_unct_r_names, unct_r[choice])
            unct_v = np.load(unct_v_names[index])
            mcdp_unct_v_names = np.append(mcdp_unct_v_names, unct_v[choice])


    np.save(os.path.join(args.in_pred_dir_summar, 'summar_label'), label_names)
    np.save(os.path.join(args.in_pred_dir_summar, 'summar_pred'), pred_names)
    np.save(os.path.join(args.in_pred_dir_summar, 'summar_unct_b'), unct_b_names)
    np.save(os.path.join(args.in_pred_dir_summar, 'summar_unct_e'), unct_e_names)
    np.save(os.path.join(args.in_pred_dir_summar, 'summar_unct_r'), unct_r_names)
    np.save(os.path.join(args.in_pred_dir_summar, 'summar_unct_v'), unct_v_names)

