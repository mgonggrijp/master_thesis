import argparse
from hesp.config.dataset_config import DATASET_CFG_DICT


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
    '--save_state',
    action='store_true',
    default=False,
    help='Whether to save the final model states.'
)


parser.add_argument(
    '--use_uw',
    action='store_true',
    default=False,
    help='Whether to use uncertainty weighting in the loss. Only works for hyperbolic embedding space geometry.'
)


parser.add_argument(
    '--train_metrics',
    action='store_true',
    default=False,
    help='Wether or not to record the metrics for the training data during training.'
)

parser.add_argument(
    '--train_stochastic',
    action='store_true',
    default=False,
    help='If to use stochastic training.'
)

parser.add_argument(
    '--data_limit',
    default=-1,
    type=int,
    help='Maxmimum number of samples that can be used. For memory preservation.'
)

parser.add_argument(
    '--warmup_epochs',
    default=0,
    type=int,
    help="The number of warmup epochs. \
         Linearly increases warmup from [1 / warmup_epochs * slr] to [slr]. "
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
    default='cuda:0',
    choices=['cpu', 'cuda:0'],
    help="Which device to use."
)

parser.add_argument(
    "--slr",
    default=0.0001,
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

