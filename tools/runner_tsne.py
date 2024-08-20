import torch

from tools import builder
from utils import misc, dist_utils, tsne_utils
from utils.logger import *

from openTSNE import TSNE
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

from datasets import data_transforms

from torchvision import transforms

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def tsne_net(args, config):
    logger = get_logger(args.log_name)
    print_log('T-Sne start ... ', logger=logger)

    # build model
    if args.ckpts is not None:
        if args.test_model == 'point_bert' \
            or args.test_model == 'maskpoint'\
            or args.test_model == 'ulip':

            config.model.encoder_dims = 256

        elif args.test_model == 'act' \
            or args.test_model == 'point_mae'\
             or args.test_model == 'point_lgmask':

            config.model.encoder_dims = 384

        elif args.test_model == 'pointgpt' \
            or args.test_model == 'point_m2ae':
            pass

        else:
            raise NotImplementedError('test_model is null or error')

        print_log(f'[test_model: ] {args.test_model}', logger='Info')
        print_log(f'[encoder_dims: ] {config.model.encoder_dims}', logger='Info')


    # build dataset
    (_, test_dataloader) =  builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)
    # load ckpts
    if args.ckpts is not None:
        builder.load_model(base_model, args.ckpts, logger=logger)
    else:
        print_log('Training from scratch', logger=logger)


    base_model = base_model.cuda()

    tsne(base_model, test_dataloader, args, config, logger=logger)


def tsne_embedding(feature):
    affinities_train = affinity.PerplexityBasedNN(
        feature,
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    init_train = initialization.pca(feature, random_state=42)

    embedding_train = TSNEEmbedding(
        init_train,
        affinities_train,
        negative_gradient_method="fft",
        n_jobs=8,
        verbose=True,
    )

    embedding_train_1 = embedding_train.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    embedding_train_2 = embedding_train_1.optimize(n_iter=500, momentum=0.8)

    return embedding_train_2


# visualization
def tsne(base_model, test_dataloader, args, config, logger=None):
    tsne = TSNE(
        perplexity=25,
        learning_rate="auto",
        metric="cosine",
        n_jobs=32,
        random_state=42,
        verbose=True,
    )

    base_model.eval()

    test_feat = []
    test_label = []
    npoints = config.npoints


    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)

            feat = base_model(points)
            test_feat.append(feat.detach())

            target = label.view(-1)
            test_label.append(target.detach())

        test_feat = torch.cat(test_feat, dim=0)
        test_feat = test_feat.cpu().numpy()

        test_label = torch.cat(test_label, dim=0)
        test_label = test_label.cpu().numpy()

        embeddings = tsne.fit(test_feat)
        tsne_utils.plot_tsne(embeddings, test_label, filename=args.tsne_fig_path)

    print("draw t-SNE ok")
