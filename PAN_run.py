import os
import argparse
import pandas as pd
import torch
import numpy as np
# from model.contrastive_embedding import contrastive_embedding_run
from data_loader import model_data_loader
from model.PAN_model import ManModel
from utils import add_dict_to_argparser
from Trainer import Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def model_defaults():
    return dict(
        embed_size=128,
        hidden_size=256,
        num_heads=1,
        dropout=0.3,
        time_num=48
    )


def create_argparser():
    defaults = dict(
        data_dir="",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.0005,
        betas=(0.9, 0.99),
        eps=1e-08,
        weight_decay=0.001,
        amsgrad=True,

        sample_num=256,
        batch_size=64,
        epochs=50,
        patience=5,
        random_seed=20,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, help='The frequency gated smoothing factor.', required=True)
    add_dict_to_argparser(parser, defaults)
    city_index = input("Please select the city(please input number): "
                       "\n\n1. New York, \n2. Tokyo, \n3. Los Angeles \n4. Houston\n")
    if city_index == '1':
        defaults2 = dict(
            dataset_city="NYC",
            cat_num=202,
        )
        add_dict_to_argparser(parser, defaults2)
    elif city_index == '2':
        defaults2 = dict(
            dataset_city="TKY",
            cat_num=184,
        )
        add_dict_to_argparser(parser, defaults2)
    elif city_index == '3':
        defaults2 = dict(
            dataset_city="LA",
            cat_num=285,
        )
        add_dict_to_argparser(parser, defaults2)
    elif city_index == '4':
        defaults2 = dict(
            dataset_city="HOU",
            cat_num=292,
        )
        add_dict_to_argparser(parser, defaults2)
    else:
        raise Exception("Invalid City Code Selected")
    return parser


def main():
    args = create_argparser().parse_args()
    # torch.manual_seed(args.random_seed)  # 为CPU设置随机种子
    # torch.cuda.manual_seed_all(args.random_seed)  # 为所有GPU设置随机种子
    train_data_iter, test_data_iter, niche_data_iter, model_kwargs = model_data_loader(args)
    poi_num = model_kwargs.get('poi_num')
    user_num = model_kwargs.get('user_num')

    if not os.path.isdir('embeddings'):
        os.mkdir('embeddings')

    poi_near_file = f'./embeddings/{args.dataset_city}_poi_contrastive_{args.embed_size}.csv'
    # poi_near_file = f'./embeddings/graphvae/{args.dataset_city}'
    if os.path.isfile(poi_near_file):
        gen_emb = torch.from_numpy(np.array(pd.read_csv(poi_near_file))).to(torch.float32)
        sos_poi = torch.rand([1, args.embed_size])
        gen_emb = torch.cat([gen_emb, sos_poi], dim=0)
    # else:
    #     gen_emb = contrastive_embedding_run(args, poi_num, model_kwargs.get('ct_dict'))
    #     gen_emb = torch.from_numpy(np.array(gen_emb)).to(torch.float32)
    #     sos_poi = torch.rand([1, args.embed_size])
    #     gen_emb = torch.cat([gen_emb, sos_poi], dim=0)

    # 定义模型及其参数
    # PAN
    model = ManModel(emb_dim=args.embed_size,
                     poi_num=poi_num,
                     hidden_dim=args.hidden_size,
                     user_num=user_num,
                     poi_emb=gen_emb,
                     cat_emb=args.cat_num+2,
                     poi_pin=model_kwargs.get('poi_freq'),
                     model_kwargs=model_kwargs,
                     drop_p=args.dropout).to(args.device)

    Trainer(args, model, model_kwargs).model_run(train_data_iter, test_data_iter, niche_data_iter)


if __name__ == '__main__':
    main()

