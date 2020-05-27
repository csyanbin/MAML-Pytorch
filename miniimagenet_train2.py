import  torch, os
import  numpy as np
from    MiniImagenet2 import MiniImagenet
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

def inf_get(train):
    while (True):
        for x in train:
            yield x

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    #np.random.seed(222)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    root = '/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch/data/miniImagenet'
    trainset = MiniImagenet(root, mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=args.imgsz)
    testset = MiniImagenet(root, mode='test', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=args.imgsz)
    trainloader = DataLoader(trainset, batch_size=args.task_num, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1, worker_init_fn=worker_init_fn, drop_last=True)
    train_data = inf_get(trainloader)
    test_data = inf_get(testloader)
    
    best_acc = 0.0 
    if not os.path.exists('ckpt/{}'.format(args.exp)):
        os.mkdir('ckpt/{}'.format(args.exp))
    for epoch in range(args.epoch):
        np.random.seed()
        x_spt, y_spt, x_qry, y_qry = train_data.__next__()
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if epoch % 100 == 0:
            print('epoch:', epoch, '\ttraining acc:', accs)

        if epoch % 2500 == 0:  # evaluation
            # save checkpoint
            torch.save(maml.state_dict(), 'ckpt/{}/model_{}.pkl'.format(args.exp,epoch))
            accs_all_test = []
            for _ in range(600):
                x_spt, y_spt, x_qry, y_qry = test_data.__next__()
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze().to(device), y_spt.squeeze().to(device), x_qry.squeeze().to(device), y_qry.squeeze().to(device)
                accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_test.append(accs)

            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            with open('ckpt/'+args.exp+'/test.txt', 'a') as f:
                print('test epoch {}: acc:{:.4f}'.format(epoch, accs[-1]),file=f)


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
    argparser.add_argument('--gpu', type=str, default='0', help="gpu ids, default:0")

    args = argparser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
