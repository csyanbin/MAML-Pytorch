import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def cal_conf(result_array):
    """result_array: nxsteps"""
    m       = np.mean(result_array, 0)
    std     = np.std(result_array, 0)
    ci95    = 1.96*std / np.sqrt(len(result_array[0]))
    
    return m,std,ci95

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch/data/miniImagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
    mini_val = MiniImagenet('/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch/data/miniImagenet/', mode='val', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=600, resize=args.imgsz)
    mini_test = MiniImagenet('/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch/data/miniImagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=600, resize=args.imgsz)

    best_acc = 0.0 
    if not os.path.exists('ckpt/{}'.format(args.exp)):
        os.mkdir('ckpt/{}'.format(args.exp))
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 500 == 0:
                print('step:', step, '\ttraining acc:', accs)
            if step % 1000 == 0:  # evaluation
                db_val  = DataLoader(mini_val, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_val = []
                for x_spt, y_spt, x_qry, y_qry in db_val:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_val.append(accs)
                mean,std,ci95 = cal_conf(np.array(accs_all_val))
                print('Val acc:{}, std:{}. ci95:{}'.format(mean[-1],std[-1],ci95[-1]))
                if mean[-1]>best_acc or step%5000==0:
                    best_acc = mean[-1]
                    torch.save(maml.state_dict(), 'ckpt/{}/model_e{}s{}_{:.4f}.pkl'.format(args.exp,epoch,step,best_acc))
                    with open('ckpt/'+args.exp+'/val.txt', 'a') as f:
                        print('val epoch {}, step {}: acc_val:{:.4f}, ci95:{:.4f}'.format(epoch,step,best_acc,ci95[-1]),file=f)
                    
                    ## Test
                    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []
                    for x_spt, y_spt, x_qry, y_qry in db_test:
                        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                        accs_all_test.append(accs)
                    mean,std,ci95 = cal_conf(np.array(accs_all_test))
                    print('Test acc:{}, std:{}, ci95:{}'.format(mean[-1], std[-1], ci95[-1]))
                    with open('ckpt/'+args.exp+'/test.txt', 'a') as f:
                        print('test epoch {}, step {}: acc_test:{:.4f}, ci95:{:.4f}'.format(epoch,step,mean[-1],ci95[-1]), file=f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--exp', type=str, help='exp string', default="exp")
    argparser.add_argument('--gpu', type=str, help='gpu id', default="0")

    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
