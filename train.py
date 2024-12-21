import time
import torch.utils.data as Data
from importlib import import_module
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import SN_Net as network
import argparse
import Dataset
import yaml
import shutil

import sys
sys.path.append("..")
from utils import *
import Loss.loss as loss

def val(net,loader_test,train_recode):
    print("eval")
    train_recode.write("======= test! ======\n")
    net.eval()
    psnr_val = 0
    ssim_val = 0
    with torch.no_grad():
        for batch_idx, (LQ, HQ) in enumerate(loader_test, 0):
            LQ = LQ.to(device)
            HQ = HQ.to(device)
            out_val = net(LQ)
            out_val = torch.clip(out_val, 0., 1.)
            psnr_val += batch_PSNR(HQ, out_val)
            ssim_val += batch_ssim(HQ, out_val)
            if batch_idx % 100 == 0:
                print("val--%d" % batch_idx)

    psnr_val /= len(dataset_val)
    ssim_val /= len(dataset_val)
    recode_test_str = "[idx %d]:PSNR_val: %.4f SSIM_val: %.4f" % ((total_idx + 1), psnr_val, ssim_val)
    print(recode_test_str)
    if args.recode:
        train_recode.write(recode_test_str)
        train_recode.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="0", help="gpu to train")
    parser.add_argument("-r", "--recode", help="choose whether to recode", action="store_true")
    parser.add_argument("-f", "--finetune", help="choose whether to finetune", action="store_true")
    parser.add_argument("-p", "--path", type=str,
                        default="",
                        help="pre weight path")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device_id = range(torch.cuda.device_count())
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # --------------train----------------------------------

    #===============加载模型===============================

    if args.finetune:
        print("finetune : {}".format(args.path))
        dir_path = args.path
        config_path = os.path.join(dir_path,"config_o.yml")
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        net = network.newton_modify(config["network"])
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=device_id)
        checkpoints = torch.load(os.path.join(dir_path,config["train"]["load"]["model"]))

        model_state_dict = checkpoints["model_state_dict"]
        current_epoch = checkpoints["current_epoch"]
        current_idx = checkpoints["current_idx"]
        optimizer_state_dict = checkpoints["optimizer_state_dict"]

        net.load_state_dict(model_state_dict)
        train_data = config["datasets"]["train"]["path"]
        val_data = config["datasets"]["test"]["path"]

        if args.recode:
            timestr = time.strftime("%Y%m%d-%H%M%S")[4:]
            recode_path = "finetune-" + timestr
            recode_path = os.path.join(dir_path, recode_path)
            os.makedirs(recode_path)
            txt_path = os.path.join(recode_path, "train_recode.txt")
            writer = SummaryWriter(log_dir=recode_path)
            train_recode = open(txt_path, 'w')
            shutil.copy(config_path, os.path.join(recode_path,"config.yml"))

    # ===============首次训练===============================

    else:
        config_path = "config.yml"
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        net = network.newton_modify(config["network"])
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=device_id)

        # net.load_state_dict(torch.load("model_real.pth"))

        train_data = config["datasets"]["train"]["path"]
        val_data = config["datasets"]["test"]["path"]

        if args.recode:
            timestr = time.strftime("%Y%m%d-%H%M%S")[4:]
            dir_path = "recode-" + config["Version"] + "-" + timestr
            dir_path = os.path.join("Recode",dir_path)
            recode_path = "First_time"
            recode_path = os.path.join(dir_path, recode_path)
            os.makedirs(recode_path)
            txt_path = os.path.join(recode_path, "train_recode.txt")
            writer = SummaryWriter(log_dir=recode_path)
            train_recode = open(txt_path, 'w')
            shutil.copy(config_path, os.path.join(dir_path, "config.yml"))
            shutil.copy(config_path, os.path.join(recode_path, "config.yml"))


    if config["train"]["optim"]["type"] == "AdamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=config["train"]["optim"]["init_lr"],
                                      weight_decay=config["train"]["optim"]["weight_decay"],betas=[0.9,0.9])
    elif config["train"]["optim"]["type"] == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config["train"]["optim"]["init_lr"],
                                      weight_decay=config["train"]["optim"]["weight_decay"], betas=[0.9, 0.9])

    if config["train"]["loss_type"] == "mse":
        loss_func = nn.MSELoss().to(device)
    elif config["train"]["loss_type"] == "mix1":
        loss_func = loss.MIX1Loss().to(device)
    elif config["train"]["loss_type"] == "mix2":
        loss_func = loss.MIX2Loss().to(device)
    elif config["train"]["loss_type"] == "Charbonnier":
        loss_func = loss.CharbonnierLoss().to(device)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset.Dataset(config["datasets"]["train"]["patch_size"],train_data,aug_mode=config["datasets"]["train"]["aug_mode"],train=True)
    dataset_val = Dataset.Dataset(0,val_data,aug_mode=2,train=False)
    loader_train = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=config["datasets"]["train"]["batch_size"], shuffle=config["datasets"]["train"]["use_shuffle"])
    loader_test = Data.DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    if (not args.finetune) or (args.finetune and not config["train"]["load"]["inherit"]):
        current_epoch = -1
        current_idx = -1
    else:
        optimizer.load_state_dict(optimizer_state_dict)

    if config["train"]["optim"]["scheduler_type"] == "linear":
        end_factor = config["train"]["optim"]["final_lr"] / config["train"]["optim"]["init_lr"]
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=len(loader_train) * config["train"]["epoch"], last_epoch=current_idx, verbose=False)
    elif config["train"]["optim"]["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_train) * config["train"]["epoch"], eta_min=config["train"]["optim"]["final_lr"],last_epoch=current_idx)

    total_idx = current_idx + 1
    last_epoch = config["train"]["epoch"] - (current_epoch + 1)

    for idx in range(last_epoch):
        epoch = idx + current_epoch + 1
        total_loss = 0
        print("-------{}------".format(epoch))
        if args.recode:
            train_recode.write("-------{}------".format(epoch))
            train_recode.write('\n')

        for batch_idx, (LQ, HQ) in enumerate(loader_train, 0):
            current_lr = round(optimizer.param_groups[0]["lr"], 7)
            net.train()
            LQ = LQ.to(device)
            HQ = HQ.to(device)

            output = net(LQ)
            optimizer.zero_grad()
            loss = loss_func(output, HQ)
            total_loss += loss.data.item()
            loss.backward()
            if config["train"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            if loss.data.item() < 1e2:
                optimizer.step()
            else:
                recode_wrong = "[idx %d][epoch %d] nan!"% (total_idx + 1,epoch + 1)
                print(recode_wrong)
                if args.recode:
                    train_recode.write(recode_wrong)
                    train_recode.write('\n')

            scheduler.step()
            total_idx += 1
            if total_idx % 10 == 0:
                recode = "[idx %d][epoch %d][%d/%d] loss: %.4f lr: %.6f" % (total_idx + 1,epoch + 1, batch_idx + 1, len(loader_train), loss.data.item(),current_lr)
                print(recode)
            if (total_idx) % config["val"]["freq"] == 0:
                val(net, loader_test, train_recode)
            if args.recode:
                writer.add_scalar('loss', loss.data.item(), total_idx)
                if total_idx % config["save"]["auto_freq"] == 0 and loss.data.item() < 1e2:
                    checkpoint = {"current_epoch" : epoch,
                                  "current_idx" : total_idx,
                                  "model_state_dict" : net.state_dict(),
                                  "optimizer_state_dict" : optimizer.state_dict()}
                    torch.save(checkpoint,os.path.join(dir_path, 'model_current.pth'))

                if total_idx % config["save"]["freq"] == 0:
                    checkpoint = {"current_epoch" : epoch,
                                  "current_idx" : total_idx,
                                  "model_state_dict" : net.state_dict(),
                                  "optimizer_state_dict" : optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(dir_path, 'model{}.pth'.format(total_idx)))

        recode = "[idx %d][epoch %d] ave_loss: %.4f" % (
        total_idx + 1, epoch + 1,  total_loss / len(loader_train))
        print(recode)
        if args.recode:
            train_recode.write(recode)
            train_recode.write('\n')

    if args.recode:
        train_recode.close()
        writer.close()
