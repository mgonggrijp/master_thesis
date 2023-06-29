import argparse
import torch
from hesp.config.config import Config
from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.models.model import model_factory
from hesp.util import data_helpers
import geoopt
from hesp.util.loss import *
import matplotlib.pyplot as plt




# region argparse
parser = argparse.ArgumentParser(
    description="Train models."
)

parser.add_argument(
    '--mode',
    type=str,
    choices=['segmenter'],
    default='segmenter',
    help="Whether to train a segmenter."
)

parser.add_argument(
    '--train_metrics',
    action='store_true',
    default=False,
    help='Wether or not to record the metrics for the training data during training.'
)

parser.add_argument(
    '--data_limit',
    default=-1,
    type=int,
    help='Maxmimum number of samples that can be used. For memory preservation.'
)


parser.add_argument(
    '--val_metrics',
    action='store_true',
    default=True,
    help='Wether or not to record the metrics for the validation data during training.'
)

parser.add_argument(
    '--base_model',
    type=str,
    choices=['deeplabv3plus'],
    default='deeplabv3plus',
    help="Choose the base model for the embedding function."
)

parser.add_argument(
    '--resume',
    action='store_true'
)

parser.add_argument(
    "--dataset",
    type=str,
    choices=DATASET_CFG_DICT.keys(),
    help="Which dataset to use",
    default='pascal',
)

parser.add_argument(
    "--geometry",
    type=str,
    choices=['euclidean', 'hyperbolic'],
    help="Type of geometry to use",
    default="hyperbolic",
)

parser.add_argument(
    "--dim",
    type=int,
    help="Dimensionality of embedding space.",
    default=256,
)

parser.add_argument(
    "--c",
    type=float,
    help="Initial curvature of hyperbolic space",
    default=1.,
)

parser.add_argument(
    "--seed",
    type=float,
    default=None,
)

parser.add_argument(
    "--flat",
    action='store_true',
    help="Disables hierarchical structure when set.",
)

parser.add_argument(
    "--freeze_bb",
    action='store_true',
    help="Freeze backbone.",
)

parser.add_argument(
    "--freeze_bn",
    action='store_true',
    help="Freeze batch normalization.",
)

parser.add_argument(
    "--batch_size",
    default=5,
    type=int,
    help="Batch size."
)

parser.add_argument(
    "--num_epochs",
    default=1,
    type=int,
    help="Number of epochs to train."
)

parser.add_argument(
    "--device",
    type=str,
    default='cuda',
    choices=['cpu', 'cuda'],
    help="Which device to use."
)

parser.add_argument(
    "--slr",
    default=0.01,
    type=float,
    help="Initial learning rate."
)

parser.add_argument(
    "--backbone",
    choices=['resnet101', 'resnet50'],
    default='resnet101',
    help="Backbone architecture."
)

parser.add_argument(
    "--output_stride",
    type=int,
    choices=[8, 16],
    default=16,
    help="Backbone output stride."
)

parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="Which GPU to use, in case of multi-gpu system and parallel training."
)

parser.add_argument(
    "--base_save_dir",
    type=str,
    default="saves",
    help="base dir used for saving models and training states."
)

parser.add_argument(
    "--segmenter_dir",
    type=str,
    default="segmenter_experiments",
    help="prefix of the directory the experiments are going to be saved."
)

parser.add_argument(
    "--zero_label",
    action='store_true',
    help='whether do zero label training.'
)

parser.add_argument(
    "--test_zero_label",
    action='store_true',
    help='whether to perform zero label testing.'
)

parser.add_argument(
    "--train",
    action='store_true',
    default=False,
    help='whether to perform testing after training.'
)

parser.add_argument(
    "--val",
    action='store_true',
    default=False,
    help='whether to perform validation after training.'
)

parser.add_argument(
    "--id",
    type=str,
    default="",
    help="Optional identifier for run"
)

parser.add_argument(
    "--json_name",
    type=str,
    default="",
    help="Needed When using ADE20K dataset. Select the hierarchy you want to use."
)

parser.add_argument(
    "--pre_trained_bb",
    action='store_true',
    default=True,
    help="If you want to use the latest pretrained backbone weights"
)

parser.add_argument(
    "--precision",
    default=32,
    type=str,
)

args = parser.parse_args()

if args.precision == '32':
    torch.set_default_dtype(torch.float32)
    
if args.precision == '64':   
    torch.set_default_dtype(torch.float64)   

config = Config(
    dataset=args.dataset,
    base_save_dir=args.base_save_dir,
    mode=args.mode,
    json_selection=args.json_name
)

config.appendix = args.id


if not args.mode:
    raise ValueError

config._IDENT = args.id
config.embedding_space._GEOMETRY = args.geometry
config.embedding_space._DIM = args.dim
config.embedding_space._INIT_CURVATURE = args.c
config.embedding_space._HIERARCHICAL = not args.flat
config.base_model_name = args.base_model
# endregion argparse


# region segmenter initialization
if args.mode == 'segmenter':
    config.segmenter._OUTPUT_STRIDE = args.output_stride
    config.segmenter._BACKBONE = args.backbone
    config.segmenter._BATCH_SIZE = args.batch_size
    config.segmenter._FREEZE_BACKBONE = args.freeze_bb
    config.segmenter._FREEZE_BN = args.freeze_bn
    config.segmenter._ZERO_LABEL = args.zero_label 
    config.segmenter._SEGMENTER_DIR = args.segmenter_dir
    config.segmenter._DEVICE = args.device
    config.segmenter._PRE_TRAINED_BB = args.pre_trained_bb
    config.segmenter._TRAIN = args.train
    config.segmenter._VAL = args.val
    config.segmenter._RESUME = args.resume
    config.segmenter._SEED = args.seed
    config.segmenter._TRAIN_METRICS = args.train_metrics
    config.segmenter._VAL_METRICS = args.val_metrics
    

    if not args.num_epochs:
        config.segmenter._NUM_EPOCHS = config.dataset._NUM_EPOCHS
    else:
        config.segmenter._NUM_EPOCHS = args.num_epochs

    if not args.slr:
        config.segmenter._INITIAL_LEARNING_RATE = config.dataset._INITIAL_LEARNING_RATE
    else:
        config.segmenter._INITIAL_LEARNING_RATE = args.slr
    config.segmenter._NUM_TRAIN = config.dataset._NUM_TRAIN
    config.segmenter._EFN_OUT_DIM = args.dim

    # endregion


# region identifier
    identifier = ""
    identifier += args.dataset 
    identifier += "_" + args.geometry
    identifier += "_dim=" + str(args.dim)
    identifier += "_c=" + str(args.c) 
    identifier += "_bs=" + str(args.batch_size)
    identifier += "_slr=" + str(args.slr) 
    identifier += "_id=" + str(args.id) if args.id else ""
    config.segmenter._SAVE_FOLDER = "saves/" + identifier + "/"
# endregion identifier


# region model and data init
    means = torch.load("datasets/" + args.dataset + "/means.pt")
    
    train_files, val_files = data_helpers.make_data_splits(
        args.dataset, limit=args.data_limit, split=(0.9, 0.1), shuffle=True)
    
    train_loader = data_helpers.make_torch_loader(
        args.dataset, train_files, config, mode='train')
    
    val_loader = data_helpers.make_torch_loader(
        args.dataset, val_files, config, mode='val')
        
    model = model_factory(config=config)
    model.identifier = identifier

    if config.segmenter._FREEZE_BN or args.freeze_bn:
        for name, param in model.embedding_model.named_parameters():
            if '.bn' in name:
                param.requires_grad = False

    learning_rate = args.slr if args.slr > 0 else config.dataset._INITIAL_LEARNING_RATE
    iters = args.num_epochs if args.num_epochs else config.dataset._NUM_EPOCHS

    optimizers = []
    schedulers = []

    # using the optimizer from geoopt that can work on manifold tensors
    params = [  {'params' : model.embedding_space.offsets, 'lr' : args.slr},
                {'params' : model.embedding_space.normals, 'lr' : args.slr},
                {'params' : model.embedding_model.classifier.parameters(), 'lr' : args.slr}]
    
    if not args.freeze_bb:
        params.append( {'params' : model.embedding_model.backbone.parameters(), 'lr' : args.slr / 10} )
    
    optimizer = geoopt.optim.rsgd.RiemannianSGD(
        params,
        lr=learning_rate,
        momentum=config.segmenter._MOMENTUM,
        weight_decay=config.segmenter._WEIGHT_DECAY,
        stabilize=None)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=iters, power=0.9, last_epoch=-1, verbose=False)
    
    print("".join([arg + ' : ' + str(args.__dict__[arg]) + "\n" for arg in args.__dict__]))
    
    from hesp.util.data_helpers import imshow
    
    for images, labels, _ in train_loader:
        imshow(images, labels)
        
    exit()
    
# endregion model and data init
    
    model.train(train_loader, val_loader, optimizer, scheduler)

