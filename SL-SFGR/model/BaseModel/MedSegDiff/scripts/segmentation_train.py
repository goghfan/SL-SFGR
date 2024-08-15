
# 导入sys模块，用于获取系统信息
import sys
# 导入argparse模块，用于解析命令行参数
import argparse
# 将guided_diffusion模块添加到系统路径中
sys.path.append("../")
sys.path.append("./")
# 导入dist_util模块，用于设置分布式训练
from guided_diffusion import dist_util, logger
# 导入resample模块，用于创建调度采样器
from guided_diffusion.resample import create_named_schedule_sampler
# 导入bratsloader模块，用于加载BRATS数据集
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
# 导入isicloader模块，用于加载ISIC数据集
from guided_diffusion.isicloader import ISICDataset
# 导入custom_dataset_loader模块，用于加载自定义数据集
from guided_diffusion.custom_dataset_loader import CustomDataset
# 导入script_util模块，用于创建模型和扩散模型
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
# 导入torch模块，用于创建tensor
import torch as th
# 导入train_util模块，用于训练模型
from guided_diffusion.train_util import TrainLoop
# 导入visdom模块，用于可视化训练结果Visdom
from visdom import Visdom
# 创建visdom实例，端口号为8850
viz = Visdom(port=8850)
# 导入torchvision.transforms模块，用于数据预处理
import torchvision.transforms as transforms

# 定义main函数，用于主函数
def main():
    # 创建argparser实例，用于解析命令行参数
    args = create_argparser().parse_args()

    # 设置分布式训练
    dist_util.setup_dist(args)
    # 配置日志
    logger.configure(dir = args.out_dir)

    # 打印日志
    logger.log("creating data loader...")

    # 根据数据集名称，选择不同的数据预处理方式
    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else :
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    # 创建数据加载器，用于加载数据集
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    # 创建数据迭代器
    data = iter(datal)

    # 打印日志
    logger.log("creating model and diffusion...")

    # 创建模型和扩散模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 设置多GPU训练
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    # 创建调度采样器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    # 打印日志
    logger.log("training...")
    # 创建训练循环
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


# 创建argparser实例，用于解析命令行参数
def create_argparser():
    # 设置默认参数
    defaults = dict(
        data_name = 'ISIC',
        data_dir="D:\\Desktop\\lung_registration\\BaseModel\\MedSegDiff\\dataset\\ISIC",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='/results/'
    )
    # 更新默认参数
    defaults.update(model_and_diffusion_defaults())
    # 创建argparser实例，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加参数到argparser实例中
    add_dict_to_argparser(parser, defaults)
    # 返回argparser实例
    return parser


# 定义main函数，用于主函数
if __name__ == "__main__":
    main()