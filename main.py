# coding=gbk 
import argparse
import os
from solver import Solver
from data_loader import *
from torch.backends import cudnn
import random


def set_random_seed(seed, deterministic=False):
    #https://www.zhihu.com/question/345043149/answer/1634128300
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
def main(config):
    #cudnn.benchmark = True
    if config.model_type not in ['My_Net','GaborNet','U_Net','EU_Net']:
        print('ERROR!! model_type should be selected in My_Net/GaborNet/U_Net/EU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    # lr = random.random()*0.0005 + 0.0000005
    lr = 0.001
    augmentation_prob= random.random()*0.7
    #augmentation_prob =0.575682354953768
    # epoch = random.choice([100,150,200,250])
    epoch = 500
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)
    #decay_epoch = 106

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr 
    config.num_epochs_decay = decay_epoch

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=1,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)
    # test_loader = get_loader_hy(image_path=config.test_path,
    #                          image_size=config.image_size,
    #                          batch_size=1,
    #                          num_workers=config.num_workers,
    #                          mode='test',
    #                          augmentation_prob=0.)


    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=256)
    parser.add_argument('--num_epochs_decay', type=int, default=106)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    #parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='My_Net', help='My_Net/GaborNet/U_Net/EU_Net')
    parser.add_argument('--model_path', type=str, default='./model_save')
    parser.add_argument('--train_path', type=str, default='./dataset03/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset03/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset03/test/')
    parser.add_argument('--result_path', type=str, default='./result_save2024')
    parser.add_argument('--cuda_idx', type=int, default=1)
    config = parser.parse_args()

    set_random_seed(10,True)
    #set_random_seed(20230223)
    main(config)

