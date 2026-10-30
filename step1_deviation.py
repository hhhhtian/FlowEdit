import glob, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("step1_outputs", exist_ok=True)
for path in glob.glob("step0_images_latent/**/*.pt", recursive=True):
    d = torch.load(path)
    x_src, x_tar, t, zt = d["x_src"].float(), d["x_tar"].float(), d["t"].float(), d["zt"].float()
    # 每个时刻的直线点
    line = (1 - t).view(-1, 1, 1, 1) * x_tar + t.view(-1, 1, 1, 1) * x_src
    # 偏差为真实路径到直线的距离除以源到目标的距离
    dev = (zt - line).flatten(1).norm(dim=1) / (x_tar - x_src).norm()
    name = os.path.relpath(path, "step0_images_latent").replace(os.sep, "_")[:-3]
    plt.figure()
    plt.plot(t, dev)
    plt.xlabel("t"); plt.ylabel(r"$\|Z^{FE}_t-[(1-t)X^{tar}+tX^{src}]\|\ /\ \|X^{tar}-X^{src}\|$")  # 纵轴公式
    plt.savefig(f"step1_outputs/{name}.png", dpi=150)
