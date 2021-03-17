import torch
from utils import *
from train_resnet import *
from denoise_ae import AutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=64, help='size of patches')
parser.add_argument('-m', '--magnitude', type=int, default=200, help='magnitude of breakHis dataset to use')
parser.add_argument('-bs', '--batch-size', type=int, default=512)
parser.add_argument('-de', '--denoising_epochs', type=int, default=50)
parser.add_argument('-re', '--resnet_epochs', type=int, default=35)
parser.add_argument('-g', '--gpus', nargs='+', type=str, default=os.environ["CUDA_VISIBLE_DEVICES"].split(sep=','),
                    help='number of GPUs to use')
parser.add_argument('-d', '--denoising', type=bool, default=True, help='number of GPUs to use')
parser.add_argument('-p', '--save_path', type=str, default='results/', help='path to save the result files')
parser.add_argument('-ld', '--load_dae', type=bool, default=True, help='load pretrained DAE')
parser.add_argument('-dp', '--dae_path', type=str, default='dae10.pt', help='path to load or save DAE')
parser.add_argument('-n', '--noise_strength', type=int, default=10)
parser.add_argument('-c', '--n_components', type=int, default=100)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus)
print('Using GPUs:', os.environ["CUDA_VISIBLE_DEVICES"])

# load data
ext = str(args.size) + '_' + str(args.magnitude) + 'X.npy'
with open('data/'+ext, 'rb') as f:
    train = np.load(f)
    valid = np.load(f)
    test = np.load(f)

train = DataSet(train[:,:-1].reshape(-1, args.size, args.size, 3).astype(float), train[:,-1].astype(np.long))
valid = DataSet(valid[:,:-1].reshape(-1, args.size, args.size, 3).astype(float), valid[:,-1].astype(np.long))
test = DataSet(test[:,:-1].reshape(-1, args.size, args.size, 3).astype(float), test[:,-1].astype(np.long))

if args.denoising:
    train_subset = train.images[::100]
    val_subset = valid.images[::100]
    model = AutoEncoder(n_inputs=train_subset.shape[1], lr=1e-3, batch_size=512, noise_strength=args.noise_strength,
                        path=args.dae_path, load_weights=False, plot=False, n_components=args.n_components)

    if not args.load_dae:
        model.fit(train_subset, Xd=val_subset, epochs=args.denoising_epochs)
       
    # apply denoising
    train = DataSet(model.transform(train.images).transpose(0,2,3,1), train.classes)
    valid = DataSet(model.transform(valid.images).transpose(0,2,3,1), valid.classes)
    test = DataSet(model.transform(test.images).transpose(0,2,3,1), test.classes)
    torch.cuda.empty_cache()
    
# train classifier
train_acc, test_acc, train_f1, test_f1 = train_resnet(train, valid, test, args.batch_size, args.resnet_epochs)
savetxt(args.save_path + str(args.size) + '_' + str(args.magnitude) + '_acc.csv',
            array([train_acc, test_acc, train_f1, test_f1]), delimiter=',')
torch.cuda.empty_cache()