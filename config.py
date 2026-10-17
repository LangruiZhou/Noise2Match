import argparse

# train
def get_config():
    parse = argparse.ArgumentParser()

    parse.add_argument("--batch_size", type=int, default=1)     # try: batch_size = 1, 2, 4
    parse.add_argument("--epoch_num", type=int, default=80)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--img_size", type=int, default=512)
    parse.add_argument("--lr", type=float, default=0.0005)

    # path
    # dataset root
    parse.add_argument("--data_root", type=str, default=r"../Data/mayo2016")  # todo
    # model weight save path
    parse.add_argument("--weight_path", type=str, default="./Result/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="./Result/weights")
    # image save path
    parse.add_argument("--train_img_root", type=str, default="./Result/train_images")

    # data norm
    parse.add_argument("--hu_min", type=float, default=-1024.)
    parse.add_argument("--hu_max", type=float, default=3072.)
    
    # loss weight
    parse.add_argument("--lambda_corr", type=float, default=1)      
    parse.add_argument("--lambda_smooth", type=float, default=0.001)    

    # noise2match
    parse.add_argument("--k", type=int, default=1)
    parse.add_argument("--th", type=float, default=80.)

    config = parse.parse_args()
    return config


