# train.py
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Function, Variable
from barbar import Bar
from sklearn.cluster import KMeans
from model import Generator, Encoder, Discriminatorxz, Discriminatorzz, Discriminatorxx, BatchClassifier, CellClassifier
from utils.utils import weights_init_normal
import numpy as np
import matplotlib.pyplot as plt
import umap
import os
import pandas as pd



def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)


class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL) with a scaling factor.
    """
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class IdentityGRL(Function):
    """
    Gradient Reversal Layer (GRL) with a scaling factor.
    """
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ALADTrainer:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader = data
        self.device = device

        # Warm-up / Loss 权重配置
        self.base_lambda_adv = getattr(self.args, "lambda_adv", 1.0)
        self.base_lambda_cycle = getattr(self.args, "lambda_cycle", 0.5)
        self.base_lambda_batch = getattr(self.args, "lambda_batch", 0.4)
        self.base_lambda_cell = getattr(self.args, "lambda_cell", 0.4)
        self.warmup_epochs = getattr(self.args, "warmup_epochs", 10)
        self.ramp_epochs = getattr(self.args, "ramp_epochs", 10)

        self.build_models()

    def _grad_norm(self, modules):
        total = 0.0
        for module in modules:
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def _lambda_schedule(self, epoch: int):

        lambda_adv = self.base_lambda_adv
        lambda_cycle = self.base_lambda_cycle

        if epoch < self.warmup_epochs:
            lambda_batch = 0.0
            lambda_cell = 0.0
        elif epoch < self.warmup_epochs + max(self.ramp_epochs, 1):
            progress = (epoch - self.warmup_epochs + 1) / max(self.ramp_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            lambda_batch = self.base_lambda_batch * progress
            lambda_cell = self.base_lambda_cell * progress
        else:
            lambda_batch = self.base_lambda_batch
            lambda_cell = self.base_lambda_cell

        return lambda_adv, lambda_cycle, lambda_batch, lambda_cell

    def train(self):
        """Training the ALAD with batch effect removal."""
        
        if self.args.pretrained:
            self.load_weights()

        optimizer_ge = optim.Adam(list(self.G.parameters()) +
                                  list(self.E.parameters()) +
                                  list(self.BatchClassifier.parameters()) +
                                  list(self.cellClassifier.parameters()),
                                  lr=self.args.lr, betas=(0.5, 0.999))
        params_ = list(self.Dxz.parameters()) \
                + list(self.Dzz.parameters()) \
                + list(self.Dxx.parameters()) 
        optimizer_d = optim.Adam(params_, lr=self.args.lr, betas=(0.5, 0.999))
        torch.manual_seed(42)
        criterion = nn.BCEWithLogitsLoss()
        batch_criterion = nn.CrossEntropyLoss()
        cell_criterion = nn.CrossEntropyLoss()

        epoch_losses = {
            "discriminator_loss": [],"generator_loss": [],"batch_loss": [],"cell_loss": [],"dxz_loss": [],"dzz_loss": [],"dxx_loss": [],"gexz_loss": [],"gezz_loss": [],"gexx_loss": [],
            "prob_dxz_real": [],"prob_dxz_fake": [],"prob_dzz_real": [],"prob_dzz_fake": [],"prob_dxx_real": [],"prob_dxx_fake": [],
            "grad_norm_G": [],"grad_norm_E": [],"grad_norm_Dxz": [],"grad_norm_Dzz": [],"grad_norm_Dxx": []
        }
        for epoch in range(self.args.num_epochs+1):
            lambda_adv, lambda_cycle, lambda_batch, lambda_cell = self._lambda_schedule(epoch)
            batch_active = lambda_batch > 0.0
            cell_active = lambda_cell > 0.0

            set_requires_grad(self.BatchClassifier, batch_active)
            set_requires_grad(self.cellClassifier, cell_active)

            d_losses, ge_losses, batch_losses, cell_losses = 0, 0, 0, 0
            d_loss_dxz, d_loss_dzz, d_loss_dxx = 0, 0, 0
            ge_loss_gexz, ge_loss_gezz, ge_loss_gexx , ge_loss_cycle = 0, 0, 0, 0

            all_z_gen = []
            all_onehot = []
            all_barcodes = []
            all_labels = []

            prob_sums = {
                "dxz_real": 0.0, "dxz_fake": 0.0,
                "dzz_real": 0.0, "dzz_fake": 0.0,
                "dxx_real": 0.0, "dxx_fake": 0.0
            }
            grad_sums = {
                "G": 0.0, "E": 0.0,
                "Dxz": 0.0, "Dzz": 0.0, "Dxx": 0.0
            }

            for x, labels, onehot, barcodes in Bar(self.train_loader):
                x = x.to(self.device)
                onehot = onehot.to(self.device)
                all_labels.extend(labels.detach().numpy())
                labels = labels.to(self.device)  

                y_true = Variable(torch.ones((x.size(0), 1)).to(self.device))
                y_fake = Variable(torch.zeros((x.size(0), 1)).to(self.device))

                z_real = Variable(torch.randn((x.size(0), self.args.latent_dim)).to(self.device), requires_grad=False)
                x_gen = self.G(z_real)

                x_real = x.float().to(self.device)
                z_gen = self.E(x_real)

                all_z_gen.append(z_gen.detach())  
                all_onehot.append(onehot.detach())
                all_barcodes.extend(barcodes)

                out_truexz, _ = self.Dxz(x_real, z_gen)
                out_fakexz, _ = self.Dxz(x_gen, z_real)

                out_truezz, _ = self.Dzz(z_real, z_real)
                out_fakezz, _ = self.Dzz(z_real, self.E(self.G(z_real)))

                out_truexx, _ = self.Dxx(x_real, x_real)
                out_fakexx, _ = self.Dxx(x_real, self.G(self.E(x_real)))

                prob_dxz_real = torch.sigmoid(out_truexz).mean().item()
                prob_dxz_fake = torch.sigmoid(out_fakexz).mean().item()
                prob_dzz_real = torch.sigmoid(out_truezz).mean().item()
                prob_dzz_fake = torch.sigmoid(out_fakezz).mean().item()
                prob_dxx_real = torch.sigmoid(out_truexx).mean().item()
                prob_dxx_fake = torch.sigmoid(out_fakexx).mean().item()

                prob_sums["dxz_real"] += prob_dxz_real
                prob_sums["dxz_fake"] += prob_dxz_fake
                prob_sums["dzz_real"] += prob_dzz_real
                prob_sums["dzz_fake"] += prob_dzz_fake
                prob_sums["dxx_real"] += prob_dxx_real
                prob_sums["dxx_fake"] += prob_dxx_fake

                if batch_active:
                    z_gen_reversed = GradientReversal.apply(z_gen)
                    batch_preds = self.BatchClassifier(z_gen_reversed)
                    batch_loss = batch_criterion(batch_preds, torch.argmax(onehot, dim=1))
                else:
                    batch_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

                if cell_active:
                    z_gen_reversed2 = IdentityGRL.apply(z_gen)
                    cell_preds = self.cellClassifier(z_gen_reversed2)
                    cell_loss = cell_criterion(cell_preds, labels) * 1
                else:
                    cell_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

                loss_dxz = criterion(out_truexz, y_true) + criterion(out_fakexz, y_fake)
                loss_dzz = criterion(out_truezz, y_true) + criterion(out_fakezz, y_fake)
                loss_dxx = criterion(out_truexx, y_true) + criterion(out_fakexx, y_fake)

                loss_d = loss_dxz + loss_dzz + loss_dxx 

                loss_gexz = criterion(out_fakexz, y_true) + criterion(out_truexz, y_fake)
                loss_gezz = criterion(out_fakezz, y_true) + criterion(out_truezz, y_fake)
                loss_gexx = criterion(out_fakexx, y_true) + criterion(out_truexx, y_fake)
                cycle_consistency = loss_gezz + loss_gexx

                loss_ge = lambda_adv * loss_gexz \
                          + lambda_cycle * cycle_consistency \
                          + lambda_batch * batch_loss \
                          + lambda_cell * cell_loss

                optimizer_d.zero_grad()
                optimizer_ge.zero_grad()

                loss_d.backward(retain_graph=True)

                grad_norm_Dxz = self._grad_norm([self.Dxz])
                grad_norm_Dzz = self._grad_norm([self.Dzz])
                grad_norm_Dxx = self._grad_norm([self.Dxx])

                grad_sums["Dxz"] += grad_norm_Dxz
                grad_sums["Dzz"] += grad_norm_Dzz
                grad_sums["Dxx"] += grad_norm_Dxx

                set_requires_grad(self.Dxz, False)
                set_requires_grad(self.Dzz, False)
                set_requires_grad(self.Dxx, False)

                loss_ge.backward()

                grad_norm_G = self._grad_norm([self.G])
                grad_norm_E = self._grad_norm([self.E])

                grad_sums["G"] += grad_norm_G
                grad_sums["E"] += grad_norm_E

                set_requires_grad(self.Dxz, True)
                set_requires_grad(self.Dzz, True)
                set_requires_grad(self.Dxx, True)

                optimizer_d.step()
                optimizer_ge.step()

                ge_losses += loss_ge.item()
                d_losses += loss_d.item()
                batch_losses += batch_loss.item()
                cell_losses += cell_loss.item()

                d_loss_dxz += loss_dxz.item()
                d_loss_dzz += loss_dzz.item()
                d_loss_dxx += loss_dxx.item()
                ge_loss_gexz += loss_gexz.item()
                ge_loss_cycle += cycle_consistency.item()
                ge_loss_gezz += loss_gezz.item()
                ge_loss_gexx += loss_gexx.item()

            num_batches = len(self.train_loader)
            prob_avg = {k: v / num_batches for k, v in prob_sums.items()}
            grad_avg = {k: v / num_batches for k, v in grad_sums.items()}

            print("Epoch: {}, Discriminator Loss: {:.3f}, Generator Loss: {:.3f}, Batch Loss: {:.3f}, Cell Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), ge_losses/len(self.train_loader), batch_losses/len(self.train_loader), cell_losses/len(self.train_loader)
            ))
            print(
            "Dxz: {:.3f}, Dzz: {:.3f}, Dxx: {:.3f}, GExz (Adversarial): {:.3f}, GEzz: {:.3f}, GExx: {:.3f}".format(
                d_loss_dxz/len(self.train_loader),
                d_loss_dzz/len(self.train_loader),
                d_loss_dxx/len(self.train_loader),
                ge_loss_gexz/len(self.train_loader),
                ge_loss_gezz/len(self.train_loader),
                ge_loss_gexx/len(self.train_loader),
            ))
            print(
                "Prob(Dxz real/fake): {:.3f}/{:.3f}, Prob(Dzz real/fake): {:.3f}/{:.3f}, Prob(Dxx real/fake): {:.3f}/{:.3f}".format(
                    prob_avg["dxz_real"], prob_avg["dxz_fake"],
                    prob_avg["dzz_real"], prob_avg["dzz_fake"],
                    prob_avg["dxx_real"], prob_avg["dxx_fake"]
                )
            )
            print(
                "Grad‖G‖: {:.3f}, Grad‖E‖: {:.3f}, Grad‖Dxz‖: {:.3f}, Grad‖Dzz‖: {:.3f}, Grad‖Dxx‖: {:.3f}".format(
                    grad_avg["G"], grad_avg["E"],
                    grad_avg["Dxz"], grad_avg["Dzz"], grad_avg["Dxx"]
                )
            )
            print(
                "λ_adv: {:.2f}, λ_cycle: {:.2f}, λ_batch: {:.2f}, λ_cell: {:.2f}".format(
                    lambda_adv, lambda_cycle, lambda_batch, lambda_cell
                )
            )

            epoch_losses["discriminator_loss"].append(d_losses / len(self.train_loader))
            epoch_losses["generator_loss"].append(ge_losses / len(self.train_loader))
            epoch_losses["batch_loss"].append(batch_losses / len(self.train_loader))
            epoch_losses["cell_loss"].append(cell_losses / len(self.train_loader))
            epoch_losses["dxz_loss"].append(d_loss_dxz / len(self.train_loader))
            epoch_losses["dzz_loss"].append(d_loss_dzz / len(self.train_loader))
            epoch_losses["dxx_loss"].append(d_loss_dxx / len(self.train_loader))
            epoch_losses["gexz_loss"].append(ge_loss_gexz / len(self.train_loader))
            epoch_losses["gezz_loss"].append(ge_loss_gezz / len(self.train_loader))
            epoch_losses["gexx_loss"].append(ge_loss_gexx / len(self.train_loader))
            epoch_losses["prob_dxz_real"].append(prob_avg["dxz_real"])
            epoch_losses["prob_dxz_fake"].append(prob_avg["dxz_fake"])
            epoch_losses["prob_dzz_real"].append(prob_avg["dzz_real"])
            epoch_losses["prob_dzz_fake"].append(prob_avg["dzz_fake"])
            epoch_losses["prob_dxx_real"].append(prob_avg["dxx_real"])
            epoch_losses["prob_dxx_fake"].append(prob_avg["dxx_fake"])
            epoch_losses["grad_norm_G"].append(grad_avg["G"])
            epoch_losses["grad_norm_E"].append(grad_avg["E"])
            epoch_losses["grad_norm_Dxz"].append(grad_avg["Dxz"])
            epoch_losses["grad_norm_Dzz"].append(grad_avg["Dzz"])
            epoch_losses["grad_norm_Dxx"].append(grad_avg["Dxx"])
        self.save_weights()

        results_dir = "loss"
        os.makedirs(results_dir, exist_ok=True)

        df_losses = pd.DataFrame(epoch_losses)
        csv_path = os.path.join(results_dir, "epoch_losses.csv")
        json_path = os.path.join(results_dir, "epoch_losses.json")

        df_losses.to_csv(csv_path, index=False)
        df_losses.to_json(json_path, orient="records", indent=2)

        print(f"[INFO] epoch_losses save:\n  - {csv_path}\n  - {json_path}")
        self.lossPlot()
        return epoch_losses

    def build_models(self):
        self.G = Generator(latent_dim=self.args.latent_dim, output_dim=self.args.dim).to(self.device)
        self.E = Encoder(latent_dim=self.args.latent_dim, input_dim=self.args.dim, do_spectral_norm=self.args.spec_norm).to(self.device)
        self.Dxz = Discriminatorxz(z_dim=self.args.latent_dim, x_dim=self.args.dim, do_spectral_norm=self.args.spec_norm).to(self.device)
        self.Dxx = Discriminatorxx(input_dim=self.args.dim, do_spectral_norm=self.args.spec_norm).to(self.device)
        self.Dzz = Discriminatorzz(input_dim=self.args.latent_dim, do_spectral_norm=self.args.spec_norm).to(self.device)

        self.BatchClassifier = BatchClassifier(latent_dim=self.args.latent_dim, num_batches=self.args.num_batches).to(self.device)
        self.cellClassifier = CellClassifier(latent_dim=self.args.latent_dim,num_celltypes=self.args.num_celltypes).to(self.device)

        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.Dxz.apply(weights_init_normal)
        self.Dxx.apply(weights_init_normal)
        self.Dzz.apply(weights_init_normal)
        self.BatchClassifier.apply(weights_init_normal)
        self.cellClassifier.apply(weights_init_normal)

    def save_weights(self):
        state_dict_Dxz = self.Dxz.state_dict()
        state_dict_Dxx = self.Dxx.state_dict()
        state_dict_Dzz = self.Dzz.state_dict()
        state_dict_E = self.E.state_dict()
        state_dict_G = self.G.state_dict()
        state_dict_BatchClassifier = self.BatchClassifier.state_dict()
        state_dict_cellClassifier = self.cellClassifier.state_dict()
        torch.save({'Generator': state_dict_G,
                    'Encoder': state_dict_E,
                    'Discriminatorxz': state_dict_Dxz,
                    'Discriminatorxx': state_dict_Dxx,
                    'Discriminatorzz': state_dict_Dzz,
                    'BatchClassifier': state_dict_BatchClassifier,
                    'cellClassifier': state_dict_cellClassifier}, 'weights/model_parameters.pth')

    def load_weights(self):
        state_dict = torch.load('weights/model_parameters.pth')

        self.Dxz.load_state_dict(state_dict['Discriminatorxz'])
        self.Dxx.load_state_dict(state_dict['Discriminatorxx'])
        self.Dzz.load_state_dict(state_dict['Discriminatorzz'])
        self.G.load_state_dict(state_dict['Generator'])
        self.E.load_state_dict(state_dict['Encoder'])
        self.BatchClassifier.load_state_dict(state_dict['BatchClassifier'])
        self.cellClassifier.load_state_dict(state_dict['cellClassifier'])


    def lossPlot(self):
        results_dir = "loss"
        csv_path = os.path.join(results_dir, "epoch_losses.csv")
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        epochs = range(len(df))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["generator_loss"], label="Generator Loss", color="#1f77b4")
        plt.plot(epochs, df["discriminator_loss"], label="Discriminator Loss", color="#ff7f0e")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator vs Discriminator Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gen_vs_disc_loss.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["batch_loss"], label="Batch Loss", color="#2ca02c")
        plt.plot(epochs, df["cell_loss"], label="Cell Loss", color="#d62728")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Batch & Cell Classification Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "batch_cell_loss.png"), dpi=300)
        plt.close()


        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["dxz_loss"], label="Dxz Loss")
        plt.plot(epochs, df["dzz_loss"], label="Dzz Loss")
        plt.plot(epochs, df["dxx_loss"], label="Dxx Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Discriminator Components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "discriminator_components.png"), dpi=300)
        plt.close()


        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["gexz_loss"], label="GExz (Adversarial)")
        plt.plot(epochs, df["gezz_loss"], label="GEzz")
        plt.plot(epochs, df["gexx_loss"], label="GExx")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator Adversarial & Cycle Terms")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "generator_components.png"), dpi=300)
        plt.close()

        print(f"[INFO] save {plots_dir}")