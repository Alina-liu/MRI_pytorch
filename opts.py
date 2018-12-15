import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')

    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help=
        'Number of classes (an: 2, tri: 3)'
    )
    '''
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    '''

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    #parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    #parser.add_argument(
     #   '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')

   # parser.add_argument(
    #    '--nesterov', action='store_true', help='Nesterov momentum')
    #parser.set_defaults(nesterov=False)
   # parser.add_argument(
    #    '--optimizer',
    #    default='sgd',
    #    type=str,
    #    help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')

    parser.add_argument(
        '--sample_size1',
        default=95,
        type=int,
        help='width of inputs')

    parser.add_argument(
        '--sample_size2',
        default=79,
        type=int,
        help='Height of inputs')

    parser.add_argument(
        '--sample_duration',
        default=79,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--mode',
        default='score',
        type=str,
        help='Mode (score | feature). score outputs class scores. '
             'feature outputs features (after global average pooling).')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')

    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--model_name',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
