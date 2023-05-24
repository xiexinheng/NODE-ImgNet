import utils.utility as utility
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import glob
from PIL import Image
import torch
import numpy as np


class ColorTrainer:
    def __init__(self, args, loader, model, loss, device, ckp):
        self.error_last = 0
        self.epoch_num = 0
        self.args = args
        self.learning_rate = args.lr
        self.model = model.to(device)
        self.optimizer = utility.make_optimizer(args, self.model)
        self.noise_level = args.noise_level
        self.patience = args.patience
        self.clearPic_loader = loader.ClearPicLoader
        self.device = device
        self.loss = loss
        self.model = model
        self.ckp = ckp
        self.validation_loss = 10 ** 5
        self.early_stopping = utility.EarlyStopping(patience=self.patience)
    def train(self):
        self.loss.step()  # step scheduler
        self.epoch_num = self.optimizer.get_last_epoch() + 1
        utility.reduce_lr_on_plateau(self.optimizer, self.epoch_num, self.loss.log, self.args)
        lr = self.optimizer.param_groups[0]['lr']
        self.loss.start_log()  # concatenate a log tensor of size = loss_types at the beginning of each epoch
        timer_data, timer_model = utility.timer(), utility.timer()
        self.ckp.write_log(
            'trainer has changed [Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch_num, lr)
        )
        for batch_count, batch_data_clean_Pic in enumerate(self.clearPic_loader):
            timer_data.hold()
            timer_model.tic()
            img_clean = batch_data_clean_Pic.to(self.device)
            if hasattr(self.args, 'is_blind') and self.args.is_blind:
                # assume imgs_clean is a batch of torch tensors representing images
                batch_size = img_clean.shape[0]
                # generate a random noise level for each image in the batch
                noise_levels = np.random.uniform(0, 55, size=batch_size)
                # add noise with the random noise level for each image in the batch
                noise = []
                for i in range(batch_size):
                    noise_one = torch.normal(0, noise_levels[i] / 255.0, img_clean[i].shape)
                    noise.append(noise_one)
                # combine the noisy images back into a batch tensor
                noise = torch.stack(noise, dim=0).to(self.device)
            else:
            # execute code if is_blind is False or the argument is not present
                noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(
                    self.device)
            img_noise = img_clean + noise
            pred_image = self.model(img_noise.to(self.device))
            loss = 0.5 * self.loss(pred_image, img_clean)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer_model.hold()

            if (batch_count + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_count + 1) * self.args.batch_size,
                    len(self.clearPic_loader.dataset),
                    self.loss.display_loss(batch_count),
                    timer_model.release(),
                    timer_data.release()))
            self.time_per100 = timer_model.ret
            timer_data.tic()

        self.loss.end_log(len(self.clearPic_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        #validation
        self.ckp.write_log('Start validation...')
        validation_loss = 0
        torch.set_grad_enabled(False)
        imgs_vali = glob.glob(os.path.join(self.args.validation_file_path, '*.png'))
        imgs_vali += glob.glob(os.path.join(self.args.validation_file_path, '*.bmp'))
        imgs_vali.sort()
        for img_vali in imgs_vali:
            img_vali = Image.open(img_vali)
            img_vali_np = np.array(img_vali)/ 255.0
            img_vali_np = np.transpose(img_vali_np, (2, 0, 1))
            img_vali_tensor = torch.from_numpy(img_vali_np)
            img_clean = img_vali_tensor.unsqueeze(0).to(self.device)
            noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(self.device)
            img_noise = img_clean + noise
            img_est = self.model(img_noise.to(self.device))
            validation_loss += 0.5 * self.loss(img_est, img_clean)
        self.validation_loss = validation_loss
        self.ckp.write_log('End validation, validation loss is:' + str(validation_loss))
        torch.set_grad_enabled(True)

    def test(self):
        torch.set_grad_enabled(False)
        epoch_num = self.optimizer.get_last_epoch()
        self.model.eval()
        self.ckp.write_log('\nStart Evaluation...')
        imgs_test = glob.glob(os.path.join(self.args.test_file_path, '*.png'))
        imgs_test += glob.glob(os.path.join(self.args.test_file_path, '*.bmp'))
        imgs_test.sort()

        self.ckp.add_log(torch.zeros(1, len(imgs_test), 2))  # create the tensor to store the loss
        val_psnr = torch.zeros(len(imgs_test))
        val_ssim = torch.zeros(len(imgs_test))
        for idx, img_test in enumerate(imgs_test):
            img_test = Image.open(img_test)
            img_test_np = np.array(img_test)/ 255.0
            img_test_np = np.transpose(img_test_np, (2, 0, 1))
            img_test_tensor = torch.from_numpy(img_test_np)
            img_clean = img_test_tensor.unsqueeze(0).to(self.device)
            noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(self.device)
            img_noise = img_clean + noise
            img_est = self.model(img_noise.to(self.device)).detach()
            img_est_cpu, img_clean_cpu = img_est.clone().cpu(), img_clean.clone().cpu()
            img_est_int, img_clean_int = (torch.clamp(torch.mul(i, 255), 0, 255).int() for i in
                                          [img_est_cpu, img_clean_cpu])
            if img_est_int.ndim == 4:
                img_est_int, img_clean_int = img_est_int.squeeze(0), img_clean_int.squeeze(0)
            img_est_np, img_clean_np = (i.permute(1, 2, 0).numpy().astype(np.uint8) for i in
                                        [img_est_int, img_clean_int])
            ssim = structural_similarity(img_est_np, img_clean_np, channel_axis=2)
            psnr = peak_signal_noise_ratio(img_clean_np, img_est_np, data_range=255)
            one_psnr, one_ssim = np.mean(psnr), np.mean(ssim)
            self.ckp.log[-1, idx, 0] = one_psnr
            self.ckp.log[-1, idx, 1] = one_ssim
            val_psnr[idx], val_ssim[idx] = one_psnr, one_ssim
            #save image
            save_list = [img_est.squeeze()]
            save_list.extend([img_noise, img_clean])
            if self.args.save_results:
                self.ckp.save_results(img_clean, idx, save_list)
            clean_image, img_est = img_clean.cpu(), img_est.cpu()
            best = self.ckp.log.max(0)

            for i, metric in enumerate(['PSNR', 'SSIM']):
                self.ckp.write_log(
                    'test{}\t{}: {:.3f} (Best: {:.3f} @epoch {})'.format(idx,metric, self.ckp.log[-1, idx, i],
                        best[0][idx, i], best[1][idx, i]
                    )
                )
        self.ckp.write_log('Saving...')
        self.CBSD68_psnr_average = torch.mean(val_psnr)
        print(self.CBSD68_psnr_average)
        self.CBSD68_ssim_average = torch.mean(val_ssim)

        if not self.args.test_only:
            self.ckp.save_model(self, epoch_num, is_best=(best[1][0, 0] + 1 == epoch_num))

        torch.set_grad_enabled(True)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            self.early_stopping(self.validation_loss)  # modified by YWU 15 July
            if self.early_stopping.early_stop:  # modified by YWU 15 July
                print("Early stopping")  # added by YWU 23 June
                return True  # added by YWU 23 June
            else:
                epoch_num = self.optimizer.get_last_epoch()
                return epoch_num >= self.args.epochs


class GrayTrainer:
    def __init__(self, args, loader, model, loss, device, ckp):
        self.error_last = 0
        self.epoch_num = 0
        self.args = args
        self.learning_rate = args.lr
        self.model = model.to(device)
        self.optimizer = utility.make_optimizer(args, self.model)
        self.noise_level = args.noise_level
        self.patience = args.patience
        self.clearPic_loader = loader.ClearPicLoader
        self.device = device
        self.loss = loss
        self.model = model
        self.ckp = ckp
        self.validation_loss = 10 ** 5
        self.early_stopping = utility.EarlyStopping(patience=self.patience)
    def train(self):
        self.loss.step()  # step scheduler
        self.epoch_num = self.optimizer.get_last_epoch() + 1
        utility.reduce_lr_on_plateau(self.optimizer, self.epoch_num, self.loss.log, self.args)
        lr = self.optimizer.param_groups[0]['lr']
        self.loss.start_log()  # concatenate a log tensor of size = loss_types at the beginning of each epoch
        timer_data, timer_model = utility.timer(), utility.timer()
        self.ckp.write_log(
            'trainer has changed [Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch_num, lr)
        )
        for batch_count, batch_data_clean_Pic in enumerate(self.clearPic_loader):
            timer_data.hold()
            timer_model.tic()
            img_clean = batch_data_clean_Pic.to(self.device)
            if hasattr(self.args, 'is_blind') and self.args.is_blind:
                # assume imgs_clean is a batch of torch tensors representing images
                batch_size = img_clean.shape[0]
                # generate a random noise level for each image in the batch
                noise_levels = np.random.uniform(0, 55, size=batch_size)
                # add noise with the random noise level for each image in the batch
                noise = []
                for i in range(batch_size):
                    noise_one = torch.normal(0, noise_levels[i] / 255.0, img_clean[i].shape)
                    noise.append(noise_one)
                # combine the noisy images back into a batch tensor
                noise = torch.stack(noise, dim=0).to(self.device)
            else:
            # execute code if is_blind is False or the argument is not present
                noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(
                    self.device)
            img_noise = img_clean + noise
            pred_image = self.model(img_noise.to(self.device))
            loss = 0.5 * self.loss(pred_image, img_clean)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer_model.hold()

            if (batch_count + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_count + 1) * self.args.batch_size,
                    len(self.clearPic_loader.dataset),
                    self.loss.display_loss(batch_count),
                    timer_model.release(),
                    timer_data.release()))
            self.time_per100 = timer_model.ret
            timer_data.tic()

        self.loss.end_log(len(self.clearPic_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        #validation
        self.ckp.write_log('Start validation...')
        validation_loss = 0
        torch.set_grad_enabled(False)
        imgs_vali = glob.glob(os.path.join(self.args.validation_file_path, '*.png'))
        imgs_vali += glob.glob(os.path.join(self.args.validation_file_path, '*.bmp'))
        imgs_vali.sort()
        for img_vali in imgs_vali:
            img_vali = Image.open(img_vali)
            img_vali = np.expand_dims(img_vali, axis=-1)
            img_vali_np = np.array(img_vali)/ 255.0
            img_vali_np = np.transpose(img_vali_np, (2, 0, 1))
            img_vali_tensor = torch.from_numpy(img_vali_np)
            img_clean = img_vali_tensor.unsqueeze(0).to(self.device)
            noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(self.device)
            img_noise = img_clean + noise
            img_est = self.model(img_noise.to(self.device))
            validation_loss += 0.5 * self.loss(img_est, img_clean)
        self.validation_loss = validation_loss
        self.ckp.write_log('End validation, validation loss is:' + str(validation_loss))
        torch.set_grad_enabled(True)

    def test(self):
        torch.set_grad_enabled(False)
        epoch_num = self.optimizer.get_last_epoch()
        self.model.eval()
        self.ckp.write_log('\nStart Evaluation...')
        imgs_test = glob.glob(os.path.join(self.args.test_file_path, '*.png'))
        imgs_test += glob.glob(os.path.join(self.args.test_file_path, '*.bmp'))
        imgs_test.sort()

        self.ckp.add_log(torch.zeros(1, len(imgs_test), 2))  # create the tensor to store the loss
        val_psnr = torch.zeros(len(imgs_test))
        val_ssim = torch.zeros(len(imgs_test))
        for idx, img_test in enumerate(imgs_test):
            img_test = Image.open(img_test)
            img_test = np.expand_dims(img_test, axis=-1)
            img_test_np = np.array(img_test)/ 255.0
            img_test_np = np.transpose(img_test_np, (2, 0, 1))
            img_test_tensor = torch.from_numpy(img_test_np)
            img_clean = img_test_tensor.unsqueeze(0).to(self.device)
            noise = torch.FloatTensor(img_clean.size()).normal_(mean=0, std=self.noise_level / 255.).to(self.device)
            img_noise = img_clean + noise
            img_est = self.model(img_noise.to(self.device)).detach()
            img_est_cpu, img_clean_cpu = img_est.clone().cpu(), img_clean.clone().cpu()
            img_est_int, img_clean_int = (torch.clamp(torch.mul(i, 255), 0, 255).int() for i in
                                          [img_est_cpu, img_clean_cpu])
            if img_est_int.ndim == 4:
                img_est_int, img_clean_int = img_est_int.squeeze(0), img_clean_int.squeeze(0)
            img_est_np, img_clean_np = (i.permute(1, 2, 0).numpy().astype(np.uint8) for i in
                                        [img_est_int, img_clean_int])
            ssim = structural_similarity(img_est_np, img_clean_np, channel_axis=2)
            psnr = peak_signal_noise_ratio(img_clean_np, img_est_np, data_range=255)
            one_psnr, one_ssim = np.mean(psnr), np.mean(ssim)
            self.ckp.log[-1, idx, 0] = one_psnr
            self.ckp.log[-1, idx, 1] = one_ssim
            val_psnr[idx], val_ssim[idx] = one_psnr, one_ssim
            #save image
            save_list = [img_est.squeeze()]
            save_list.extend([img_noise, img_clean])
            if self.args.save_results:
                self.ckp.save_results(img_clean, idx, save_list)
            clean_image, img_est = img_clean.cpu(), img_est.cpu()
            best = self.ckp.log.max(0)

            for i, metric in enumerate(['PSNR', 'SSIM']):
                self.ckp.write_log(
                    'test{}\t{}: {:.3f} (Best: {:.3f} @epoch {})'.format(idx,metric, self.ckp.log[-1, idx, i],
                        best[0][idx, i], best[1][idx, i]
                    )
                )
        self.ckp.write_log('Saving...')
        self.CBSD68_psnr_average = torch.mean(val_psnr)
        print(self.CBSD68_psnr_average)
        self.CBSD68_ssim_average = torch.mean(val_ssim)

        if not self.args.test_only:
            self.ckp.save_model(self, epoch_num, is_best=(best[1][0, 0] + 1 == epoch_num))

        torch.set_grad_enabled(True)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            self.early_stopping(self.validation_loss)  # modified by YWU 15 July
            if self.early_stopping.early_stop:  # modified by YWU 15 July
                print("Early stopping")  # added by YWU 23 June
                return True  # added by YWU 23 June
            else:
                epoch_num = self.optimizer.get_last_epoch()
                return epoch_num >= self.args.epochs
