import argparse
import os
import torch
from base_model import train_test
import random
import numpy as np

fix_seed = 0
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if __name__ == '__main__':

parser = argparse.ArgumentParser(description='Time Series Forecasting With Transformers')

# basic config
parser.add_argument('--model', type=str, required=False, default='Transformer')

# data loader
parser.add_argument('--data_path', type=str, default='./data/multivariate.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--encoder_input_len', type=int, default=96, help='input sequence length - encoder input length')
parser.add_argument('--decoder_input_len', type=int, default=48, help='start token length - decoder input length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')


# Formers
parser.add_argument('--positional_embedding', default='True')
parser.add_argument('--value_embedding', default='True')
parser.add_argument('--temporal_embedding', default='False')

parser.add_argument('--enc_in', type=int, default=20, help='encoder input size') # number depends on the features (including target feature)
parser.add_argument('--dec_in', type=int, default=20, help='decoder input size') # number depends on the features (including target feature)
parser.add_argument('--c_out', type=int, default=1, help='output size') # number depends on the features to be predicted.
parser.add_argument('--d_model', type=int, default=512, help='dimension of model') #d_model
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') #number_of_heads
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') #encoder_stacks
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') #decoder_stacks
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') ## used in the seriec decomposition block of autoformer ##

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print("Using GPU") if args.use_gpu else print("Using CPU")
print('Args in experiment:')
print(args)


model = train_test
model_setting = '{}_sl-{}_ll-{}_pl-{}_df{}_pos-{}_val-{}_temp-{}'.format(
    args.model,
    args.encoder_input_len,
    args.decoder_input_len,
    args.pred_len,
    args.d_ff,
    args.positional_embedding,
    args.value_embedding,
    args.temporal_embedding
)

Model = model(args)  # set experiments
print('Training: ' + model_setting)
Model.train(model_setting)

print('Testing')
Model.test(model_setting)
torch.cuda.empty_cache()
