import os
import argparse
from pathlib import Path
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        default='/home/Mamba3D/cfgs/finetune_scan_hardest.yaml',
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32) #8
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='flops_debug', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--svm',
        action='store_true',
        default=False,
        help='svm')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mem_cpu', 
        action='store_true', 
        default=False, 
        help = 'mem cpu finetune')
    parser.add_argument(
        '--fewshot_model', 
        action='store_true', 
        default=False, 
        help = 'fewshot modelnet')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=5)
    parser.add_argument(
        '--shot', type=int, default=10)
    parser.add_argument(
        '--fold', type=int, default=9)
    parser.add_argument(
        '--tsne_model', 
        action='store_true', 
        default=False, 
        help = 'tsne')
    parser.add_argument('--test_model', type = str, default='point_mae', help = 'tsne model name')
    
    parser.add_argument('--tsne_fig_path', type = str, default='/home/Mamba3D/tsne_pointmae_hardest_pt.pdf', help = 'tsne_fig_path name')
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'finetune training from scratch')
        
    if args.fewshot_model and args.ckpts is None:
        print(
            'fewshot training from scratch')
    
    if args.scratch_model:
        print('training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    copy_file_to_log(args, name='Mamba3D.py')
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

def copy_file_to_log(args, name):
    if not os.path.exists(os.path.join(args.experiment_path, name)):
        shutil.copyfile(os.path.join('/home/Mamba3D/models', name), os.path.join(args.experiment_path, name))
        print(f"Save {name} to {args.experiment_path}!")
    else:
        os.remove(os.path.join(args.experiment_path, name)) # delete existing file first
        print("Delete done!" if not os.path.exists(os.path.join(args.experiment_path, name)) else "Already existing!") 
        shutil.copyfile(os.path.join('/home/Mamba3D/models', name), os.path.join(args.experiment_path, name))
        print(f"Overwrite {name} to {args.experiment_path}!")
