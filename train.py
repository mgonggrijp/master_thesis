import torch
from hesp.config.config import Config
from hesp.models.model import model_factory
from hesp.util import data_helpers
import geoopt
from train_args import args

CYCLIC = True

# change according to your system
ROOT =  '/home/mgonggri/master_thesis/'

torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(sci_mode=False)

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

config._ROOT = ROOT

if not args.mode:
    raise ValueError

config._IDENT = args.id
config.embedding_space._GEOMETRY = args.geometry
config.embedding_space._DIM = args.dim
config.embedding_space._INIT_CURVATURE = args.c
config.embedding_space._HIERARCHICAL = not args.flat
config.base_model_name = args.base_model

# region segmenter initialization
if args.mode == 'segmenter':
    config.segmenter._OUTPUT_STRIDE = args.output_stride
    config.segmenter._BACKBONE = args.backbone
    config.segmenter._BATCH_SIZE = args.batch_size
    config.segmenter._FREEZE_BACKBONE = args.freeze_bb
    config.segmenter._FREEZE_BN = args.freeze_bn
    config.segmenter._SEGMENTER_DIR = args.segmenter_dir
    config.segmenter._DEVICE = args.device
    config.segmenter._PRE_TRAINED_BB = args.pre_trained_bb
    config.segmenter._TRAIN = args.train
    config.segmenter._VAL = args.val
    config.segmenter._RESUME = args.resume
    config.segmenter._SEED = args.seed
    config.segmenter._TRAIN_METRICS = args.train_metrics
    config.segmenter._VAL_METRICS = args.val_metrics
    config.segmenter._TRAIN_STOCHASTIC = args.train_stochastic
    config.segmenter._SAVE_STATE = args.save_state
    config.segmenter._USE_UNCERTAINTY_WEIGHTS = args.use_uw
    
    if not args.num_epochs:
        config.segmenter._NUM_EPOCHS = config.dataset._NUM_EPOCHS
    else:
        config.segmenter._NUM_EPOCHS = args.num_epochs

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
    config.segmenter._SAVE_FOLDER = ROOT + "saves/" + identifier + "/"
# endregion identifier

# region model and data ini
    means = torch.load(ROOT + "datasets/" + args.dataset + "/means.pt")
    train_files, val_files = data_helpers.make_data_splits(
        args.dataset,
        limit=args.data_limit,
        split=(0.8, 0.2),
        shuffle=True,
        seed=args.seed)
    
    print("[number of training samples]    {}".format( str(len(train_files) )))
    print("[number of validation samples]  {}".format( str(len(val_files)) ) )
    
    if args.train_stochastic:
        train_dataset = data_helpers.PascalDataset(
                    train_files,
                    output_size=(config.segmenter._HEIGHT, config.segmenter._WIDTH),
                    scale=(config.segmenter._MIN_SCALE, config.segmenter._MAX_SCALE))
        # set dataset into training mode and give it the means
        train_dataset.means = torch.load(ROOT + 'datasets/pascal/means.pt')
        train_dataset.use_transforms = True
        train_dataset.is_training = True
    
    else:
        train_loader = data_helpers.make_torch_loader(
            args.dataset, train_files, config, mode='train')
        
    val_loader = data_helpers.make_torch_loader(
        args.dataset, val_files, config, mode='val')\
        
    model = model_factory(config=config).to(args.device)
    
    model.identifier = identifier
    
    backbone_params = {"params" : model.embedding_model.backbone.parameters(), "lr" : args.slr/10}
    classifier_params = {"params" : model.embedding_model.classifier.parameters(), "lr" : args.slr}
    emb_space_params = {"params" : model.embedding_space.parameters(), "lr" : args.slr}
    parameters = [backbone_params, classifier_params, emb_space_params]    

    optimizer = geoopt.optim.RiemannianSGD(
        parameters,
        lr=args.slr,
        momentum=config.segmenter._MOMENTUM,
        weight_decay=config.segmenter._WEIGHT_DECAY,
        stabilize=1)
    
    model.config.segmenter._CYCLIC = CYCLIC
    
    if CYCLIC:
        print('[Training with cyclic learning rate scheduling...]')
        steps_per_epoch = len(train_files) // args.batch_size
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            args.slr,
            args.slr * 10, 
            step_size_up=2 * steps_per_epoch,
            step_size_down=None,
            mode='triangular', 
            gamma=1.0, 
            scale_fn=None, 
            scale_mode='cycle', 
            cycle_momentum=True, 
            base_momentum=0.8, 
            max_momentum=0.9,
            last_epoch=-1, 
            verbose=False)
    else:
        print('[Training with polynomial decay learning rate scheduling...]')
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=args.num_epochs, power=0.9, last_epoch=-1, verbose=False)     
    
    print("".join([arg + ' : ' + str(args.__dict__[arg]) + "\n" for arg in args.__dict__]))
    
# endregion model and data init
    # train using a stochastic method
    if args.train_stochastic:
        print('[Training with stochastic batching...]')
            
        model.train_fn_stochastic(
            train_dataset, val_loader, optimizer, scheduler)

    # train with standard dataloading, including shuffling        
    else:
        print('[Training with default shuffled batching...]')
        model.train_fn(
            train_loader, val_loader, optimizer, scheduler)

