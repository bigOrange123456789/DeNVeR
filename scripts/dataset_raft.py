import os
import subprocess

from concurrent import futures

BASE_DIR = os.path.abspath("__file__/..")
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys
sys.path.append(ROOT_DIR)
from data import get_data_subdir, match_custom_seq


def process_sequence(gpu, dtype, root, seq, gap, res="480p", batch_size=4):
    '''
        gpu:0
        dtype:"custom"
        root:"../custom_videos/"
        seq:"#CVAI-2828RAO2_CRA32"
        gap
        res:"480p"
        batch_size:4
    '''
    if dtype == "fbms":
        rgb_name = ""
    elif dtype == "custom":
        rgb_name = "PNGImages"
        seq = match_custom_seq(root, rgb_name, seq)
    elif dtype == "davis" or dtype == "stv2":
        rgb_name = "JPEGImages"
    else:
        raise NotImplementedError

    # gpu = gpu
    subds = [rgb_name, "raw_flows_gap{}".format(gap), "flow_imgs_gap{}".format(gap)]
    # subds: ['PNGImages', 'raw_flows_gap-1', 'flow_imgs_gap-1']

    rgb, out, out_img = [get_data_subdir(dtype, root, sd, seq, res) for sd in subds]
    print("rgb, out, out_img:",rgb, out, out_img) # 输入, 光流原文件, 光流图片
    # ../custom_videos/PNGImages/CVAI-2828RAO2_CRA32
    # ../custom_videos/raw_flows_gap1/CVAI-2828RAO2_CRA32 # 这个路径第二次执行为：raw_flows_gap-1
    # ../custom_videos/flow_imgs_gap1/CVAI-2828RAO2_CRA32 # out与out_img是相同的

    exe = os.path.join(BASE_DIR, "run_raft.py")
    cmd = f"python {exe} {rgb} {out} -I {out_img} --gap {gap} -b {batch_size}"
    # cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True)


def main(args):
    # print("Have you updated the path to the RAFT repo? (y/n)")
    # resp = input()
    # if resp.lower() != "y":
    #     print("Please modify scripts/run_raft.py")
    #     sys.exit()
    # print("Comment this out in scripts/run_raft.py")

    if args.root is None:
        print("Have you updated the paths to your data? (y/n)")
        resp = input()
        if resp.lower() != "y":
            print("Please modify scripts/dataset_raft.py")
            sys.exit()
        print("Comment this out in scripts/dataset_raft.py")

        if args.dtype == "fbms":
            args.root = "/path/to/FBMS_Testset"
        elif args.dtype == "davis":
            args.root = "/path/to/DAVIS"
        elif args.dtype == "stv2":
            args.root = "/path/to/SegTrackv2"
        elif args.dtype == "custom":
            args.root = "/path/to/custom_videos"

    if args.seqs is None:
        if args.dtype == "fbms":
            args.seqs = os.listdir(args.root)
        elif args.dtype == "davis":
            args.seqs = os.listdir(os.path.join(args.root, "JPEGImages", args.dres))
        elif args.dtype == "stv2":
            args.seqs = os.listdir(os.path.join(args.root, "JPEGImages"))
        elif args.dtype == "custom":
            args.seqs = os.listdir(os.path.join(args.root, "PNGImages"))
        else:
            raise NotImplementedError

    i = 0
    with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:#只有一个编号为0的GPU
        '''
        concurrent.futures:用于异步执行可调用对象（如函数）。它提供了两种类型的执行器（Executor）：
            ThreadPoolExecutor：使用线程池执行任务。
            ProcessPoolExecutor：使用进程池执行任务。
        futures.ProcessPoolExecutor:
            是一个用于并行执行任务的类。它通过创建多个进程来并行处理任务，适用于CPU密集型任务（如计算密集型操作）。
        max_workers：
            这是ProcessPoolExecutor的参数，指定了进程池中允许的最大工作进程数。
            在这行代码中，max_workers的值是len(args.gpus)，表示进程池中的工作进程数等于args.gpus的长度。
        '''
        for seq in args.seqs: #逐个处理每一段视频
            for gap in [args.gap, -args.gap]:#这个循环执行两次，一次gap=-1,另一次gap=1
                gpu = args.gpus[i % len(args.gpus)] #因为只有一个设备，所以这里获取到的编号ID始终为0
                ex.submit(
                    process_sequence, #<function process_sequence at 0x7f9c903f30d0>
                    gpu, #0
                    args.dtype, #custom
                    args.root,  #../custom_videos/
                    seq, #CVAI-2828RAO2_CRA32
                    gap,
                    args.dres, #480p
                    args.batch_size, #4
                )
                i += 1
                print("程序中断位置：[./scripts, dataset_raft.py, main(args)]")
                exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="path to dataset root folder")
    parser.add_argument(
        "--dtype", default="custom", choices=["custom", "davis", "fbms", "stv2"]
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="+", default=[0])
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--dres", default="480p")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    main(args)
