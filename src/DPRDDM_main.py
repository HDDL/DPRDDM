# -*- coding: utf-8 -*-
"""
Created on 2023/07/10 10:49

@author: Zhenjie Zheng, CEE, PolyU
"""

# -*- coding: utf-8 -*-
"""
This is an un-supervised deep learning model used to remove the irregular noises in traffic speed data.
The name of the model is the Doulbly Physics-Regularized Denoising Diffusion Model (DPRDDM).
The input of the model is the noisy speed matrix.
Our model then outputs the denoising results.
The traffic fundamental diagram is used to ensure the reocvered traffic speed conforms to the traffic dynamics.
In addition, we propose the novel physics-regularzied masks that are automatically derived from the fundamental diagram and noisy speed matrix to filter the possible noisy data.
A good property of our model is tha we do not require specific mathmatical formulations of the fundamental diagram.
An unsatisfactory property of our model is that we need a lot of time to train the model and sample the results.
"""


import torch
import DPRDDM_body as df
import numpy as np
import os
from torch import nn
import math


def import_data(data_folder):
    speed_data = []
    file_names = os.listdir(data_folder)
    file_names.sort()
    for file_name in file_names:
        if file_name[-4:] == '.txt':
            file_name = data_folder + file_name
            temp_data = np.loadtxt(file_name, delimiter=',')
            speed_data.append(temp_data)
    return speed_data

def import_recovered_data(data_folder):
    speed_data = []
    file_names = os.listdir(data_folder)
    file_names.sort()
    for file_name in file_names:
        if file_name[-4:] == '.txt':
            file_name = data_folder + file_name
            temp_data = np.loadtxt(file_name, delimiter='\t')
            speed_data.append(temp_data)
    return speed_data


def obtain_file_paths(data_folder):
    file_names = os.listdir(data_folder)
    file_names.sort()
    file_paths = []
    for temp_name in file_names:
        if temp_name[-4:] == '.txt':
            file_path = data_folder + temp_name
            file_paths.append(file_path)
    return file_paths


def output_recovered_data(recovered_data, save_folder):
    number_of_pic = recovered_data.shape[0]
    for i in range(0, number_of_pic):
        temp_data = recovered_data[i]
        if i < 9:
            file_name = save_folder + '0000' + str(i + 1) + '.txt'
        elif i < 99:
            file_name = save_folder + '000' + str(i + 1) + '.txt'
        elif i < 999:
            file_name = save_folder + '00' + str(i + 1) + '.txt'
        elif i < 9999:
            file_name = save_folder + '0' + str(i + 1) + '.txt'
        else:
            file_name = save_folder + str(i + 1) + '.txt'
        np.savetxt(file_name, temp_data, delimiter='\t')


class PCLoss(nn.Module):
    def __init__(self, VC=30, alpha=100, beta=10, gamma=10, margin=0.1, factor=0.5, gt_speed_matrix=0, count=0, save_folder='', batch_size=16, **kwargs):
        super(PCLoss, self).__init__(**kwargs)
        self.VC = VC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.factor = factor
        self.count = count
        self.save_folder = save_folder
        self.report_index = 160
        self.max_speed = 100
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        self.batch_size = batch_size
        self.all_weight_matrix = np.zeros((train_number, 1, matrix_size, matrix_size))

    def update_weight_matrix(self, last_weight_matrix, input_with_noise, pred_speed_matrix, VC, factor,
                             current_update_count):
        last_noise_index = np.argwhere(last_weight_matrix < 0.2)  # identify the noise of last
        input_with_noise = input_with_noise.cpu().numpy()
        pred_speed_matrix = pred_speed_matrix.cpu().detach().numpy()
        input_with_noise[tuple(last_noise_index.T)] = pred_speed_matrix[tuple(last_noise_index.T)]
        current_weight_matrix = self.get_weight_matrix(input_with_noise, VC, factor, masked_percent_per_update,
                                                       current_update_count)
        update_index = current_weight_matrix < last_weight_matrix
        last_weight_matrix[update_index] = 0
        current_weight_matrix = last_weight_matrix

        return current_weight_matrix

    def get_weight_matrix(self, speed_matrix, VC, factor, masked_percent_per_update, current_update_count):

        # break when the masked_percent is larger than the max_masked_percent
        # masked_percent = masked_percent_per_update * current_update_count
        # if masked_percent > max_masked_percent * 100:
        #    return np.array([0])
        input_shape = speed_matrix.shape

        # consecutive loss
        # temporal consecutive
        temporal_anchor_matrix = speed_matrix[:, :, 1:-1, :]
        temporal_last_matrix = speed_matrix[:, :, 0:-2, :]
        temporal_next_matrix = speed_matrix[:, :, 2:, :]
        temporal_loss_1 = np.square(temporal_anchor_matrix - temporal_last_matrix)
        temporal_loss_2 = np.square(temporal_anchor_matrix - temporal_next_matrix)
        temporal_loss = temporal_loss_1 + temporal_loss_2

        # spatial consecutive
        spatial_anchor_matrix = speed_matrix[:, :, :, 1:-1]
        spatial_left_matrix = speed_matrix[:, :, :, 0:-2]
        spatial_right_matrix = speed_matrix[:, :, :, 2:]
        spatial_loss_1 = np.square(spatial_anchor_matrix - spatial_left_matrix)
        spatial_loss_2 = np.square(spatial_anchor_matrix - spatial_right_matrix)
        spatial_loss = np.add(spatial_loss_1, spatial_loss_2)

        # dynamic loss
        # get anchor, last and upstream matrix first
        anchor_matrix = speed_matrix[:, :, 1:, :-1]
        last_matrix = speed_matrix[:, :, :-1, :-1]
        upstream_matrix = speed_matrix[:, :, :-1, 1:]

        # for elements which are greater than VC
        IG_l = np.greater(last_matrix, VC)
        IG_u = np.greater(upstream_matrix, VC)
        IG = np.equal(IG_l, IG_u)
        IG = np.array(IG).astype(float)
        free_flow_loss = - (upstream_matrix - last_matrix) * (anchor_matrix - last_matrix)
        free_flow_loss = np.maximum(free_flow_loss, 0)
        free_flow_loss = free_flow_loss * IG

        # for elements which are not greater than VC
        IL_l = np.less(last_matrix, VC)
        IL_u = np.less(upstream_matrix, VC)
        IL = np.equal(IL_l, IL_u)
        IL = np.array(IL).astype(float)
        congested_loss = -(upstream_matrix - last_matrix) * (last_matrix - anchor_matrix)
        congested_loss = np.maximum(congested_loss, 0)
        congested_loss = congested_loss * IL

        # merge the free flow and congested loss
        PC_loss = free_flow_loss + congested_loss
        PC_mean = np.mean(PC_loss)

        # calculate the weight matrix according to the dynamic loss
        PC_matrix = np.ones(input_shape) * PC_mean  # full the empty cell using the mean value of the dynamic loss
        PC_matrix[:, :, 1:, :-1] = PC_loss
        PC_matrix = 1 - (PC_matrix / np.max(PC_matrix))

        # calculate the weight matrix according to the spatial and temporal loss
        spatial_mean = np.mean(spatial_loss)
        spatial_matrix = np.ones(input_shape) * spatial_mean
        spatial_matrix[:, :, :, 1:-1] = spatial_loss
        # spatial_matrix = 1 - (spatial_matrix / np.max(spatial_matrix))
        # spatial_matrix = 1 - 1 / (1 + np.exp(-15 * spatial_matrix))

        temporal_mean = np.mean(temporal_loss)
        temporal_matrix = np.ones(input_shape) * temporal_mean
        temporal_matrix[:, :, 1:-1, :] = temporal_loss
        # temporal_matrix = 1 - (temporal_matrix / np.max(temporal_matrix))
        # temporal_matrix = 1 - 1 / (1 + np.exp(-15 * temporal_matrix))

        consecutive_matrix = spatial_matrix + temporal_matrix
        consecutive_matrix = 1 - (consecutive_matrix / np.max(consecutive_matrix))

        physical_matrix = factor * consecutive_matrix + (1 - factor) * PC_matrix
        physical_noise_percentile = np.percentile(physical_matrix, masked_percent_per_update)
        physical_matrix[physical_matrix <= physical_noise_percentile] = 0
        physical_matrix[physical_matrix > physical_noise_percentile] = 1
        # noise_percentile = np.percentile(consecutive_matrix, masked_percent)
        # consecutive_matrix[consecutive_matrix < noise_percentile] = 0
        # consecutive_matrix[consecutive_matrix > noise_percentile] = 1
        # PC_noise_percentile = np.percentile(PC_matrix, masked_percent)
        # PC_matrix[PC_matrix < PC_noise_percentile] = 0
        # PC_matrix[PC_matrix > PC_noise_percentile] = 1
        # weight_matrix = factor * consecutive_matrix + (1 - factor) * PC_matrix

        return physical_matrix

    def take_matrix_with_idx(self, all_weight_matrix, idx):
        matrix_height = all_weight_matrix.shape[2]
        matrix_width = all_weight_matrix.shape[3]
        weight_matrix = np.zeros((idx.size, 1, matrix_height, matrix_width))
        for i in range(0, idx.size):
            temp_idx = idx[i]
            weight_matrix[i, :, :, :] = all_weight_matrix[temp_idx, :, :, :]

        return weight_matrix

    def replace_matrix_with_idx(self, all_weight_matrix, idx, weight_matrix):
        for i in range(0, idx.size):
            temp_idx = idx[i]
            all_weight_matrix[temp_idx, :, :, :] = weight_matrix[i, :, :, :]

        return all_weight_matrix

    def output_current_result(self, current_epoch, result, save_folder):
        result = result.cpu().detach().numpy()
        result = (result + 1) * 0.5
        result = result * self.max_speed
        current_epoch = int(current_epoch)
        if current_epoch < 10:
            current_epoch = '000' + str(current_epoch)
        elif current_epoch < 100:
            current_epoch = '00' + str(current_epoch)
        elif current_epoch < 1000:
            current_epoch = '0' + str(current_epoch)
        else:
            current_epoch = str(current_epoch)
        save_file_path = save_folder + 'speed_matrix_epoch_' + current_epoch + '.txt'
        np.savetxt(save_file_path, result, delimiter='\t')

    def output_curren(self, current_epoch, result, save_folder):
        result = result.cpu().detach().numpy()
        result = (result + 1) * 0.5
        result = result * self.max_speed
        current_epoch = int(current_epoch)
        if current_epoch < 10:
            current_epoch = '000' + str(current_epoch)
        elif current_epoch < 100:
            current_epoch = '00' + str(current_epoch)
        elif current_epoch < 1000:
            current_epoch = '0' + str(current_epoch)
        else:
            current_epoch = str(current_epoch)
        save_file_path = save_folder + 'speed_matrix_epoch_' + current_epoch + '.txt'
        np.savetxt(save_file_path, result, delimiter='\t')

    def report_updated_weight_matrix(self, current_update_count, report_index):
        # report the all_weight_matrix[10,:,:,0] to check the changes of weight_matrix
        temp_weight_matrix = self.all_weight_matrix[report_index, 0, :, :]
        if current_update_count < 10:
            current_update_count = '000' + str(current_update_count)
        elif current_update_count < 100:
            current_update_count = '00' + str(current_update_count)
        elif current_update_count < 1000:
            current_update_count = '0' + str(current_update_count)
        else:
            current_update_count = str(current_update_count)
        np.savetxt(self.save_folder + 'update_' + current_update_count + '_weight_matrix.txt', temp_weight_matrix,
                   delimiter='\t')

    def forward(self, x_start, pred_x_start, input_noise, output_noise, idx, current_t, t_decay, trained_model):

        VC = self.VC
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        margin = self.margin
        margin = torch.Tensor([margin]).cuda()
        factor = self.factor
        save_folder = self.save_folder
        current_epoch = self.count // train_number_per_epoch
        current_batch = int(self.count % train_number_per_epoch)
        str_current_epoch = int(current_epoch)
        # t_decay
        t_decay = torch.unsqueeze(t_decay, 1)
        t_decay = torch.unsqueeze(t_decay, 1)
        t_decay = torch.unsqueeze(t_decay, 1)
        # idx of the inputs
        idx = idx.cpu().numpy()
        # calculate and update the weight_matrix
        if current_epoch == 0:  # for the first epoch, we can calculate the weight_matrix using the input
            current_update_count = 1
            x_start_numpy = x_start.cpu().numpy()
            weight_matrix = self.get_weight_matrix(x_start_numpy, VC, factor, masked_percent_per_update,
                                                   current_update_count)
            self.all_weight_matrix = self.replace_matrix_with_idx(self.all_weight_matrix, idx, weight_matrix)
            if current_batch == 10:
                self.report_updated_weight_matrix(current_update_count, 160)
        else:
            if current_epoch % number_of_epochs_per_update == 0:  # update the weight matrix when the current_epoch = n* number_of
                current_update_count = int(current_epoch / number_of_epochs_per_update)
                current_update_count += 1  # the first count is 1
                current_weight_matrix = self.take_matrix_with_idx(self.all_weight_matrix,idx)  # take the weight matrix with idx
                if 1 - np.sum(current_weight_matrix) / current_weight_matrix.size < max_masked_percent:  # when the noise percent is less than the max_masked_percent
                    print(-1)
                    trained_model.ema.ema_model.eval()
                    with torch.no_grad():
                        sample_x_start = trained_model.ema.ema_model.p_sample_loop_x0(x_start, save_folder='')
                        weight_matrix = self.update_weight_matrix(current_weight_matrix, x_start, sample_x_start, VC,
                                                                  factor, current_update_count, current_batch)
                        self.all_weight_matrix = self.replace_matrix_with_idx(self.all_weight_matrix, idx,
                                                                              weight_matrix)

                        if current_batch == 0:
                            self.output_current_result(current_update_count, sample_x_start[0, 0, :, :], save_folder)
                else:
                    weight_matrix = self.take_matrix_with_idx(self.all_weight_matrix, idx)
                    # report the updated weight_matrix
                    # output the update information
                if current_batch == 0:
                    self.report_updated_weight_matrix(current_update_count, idx[0])
                    # self.output_current_pred_result(current_update_count, pred_x_start[0, 0, :, :], save_folder)

                # report the updated weight_matrix

            else:
                weight_matrix = self.take_matrix_with_idx(self.all_weight_matrix,
                                                          idx)  # take the weight matrix with idx
        weight_matrix = torch.tensor(weight_matrix).cuda()  # transform the data type

        # consecutive constraint of the matrix
        # temporal consecutive
        temporal_anchor_matrix = pred_x_start[:, :, 1:-1, :]
        temporal_last_matrix = pred_x_start[:, :, 0:-2, :]
        temporal_next_matrix = pred_x_start[:, :, 2:, :]
        temporal_loss_1 = torch.square(temporal_anchor_matrix - temporal_last_matrix)
        temporal_loss_2 = torch.square(temporal_anchor_matrix - temporal_next_matrix)
        temporal_loss = torch.add(temporal_loss_1, temporal_loss_2)
        consecutive_margin = torch.Tensor([0.04]).cuda()
        temporal_loss = torch.maximum(consecutive_margin, temporal_loss)
        origin_temporal_loss = torch.mean(temporal_loss)
        temporal_loss = t_decay * temporal_loss
        temporal_loss = torch.mean(temporal_loss)

        # spatial consecutive
        spatial_anchor_matrix = pred_x_start[:, :, :, 1:-1]
        spatial_left_matrix = pred_x_start[:, :, :, 0:-2]
        spatial_right_matrix = pred_x_start[:, :, :, 2:]
        spatial_loss_1 = torch.square(spatial_anchor_matrix - spatial_left_matrix)
        spatial_loss_2 = torch.square(spatial_anchor_matrix - spatial_right_matrix)
        spatial_loss = torch.add(spatial_loss_1, spatial_loss_2)
        spatial_loss = torch.maximum(consecutive_margin, spatial_loss)
        origin_spatial_loss = torch.mean(spatial_loss)
        spatial_loss = t_decay * spatial_loss
        spatial_loss = torch.mean(spatial_loss)

        # sum of the consecutive_loss
        Consecutive_loss = (spatial_loss + temporal_loss)

        # the dynamic constraints
        # get anchor, last and upstream matrix first
        anchor_matrix = pred_x_start[:, :, 1:, :-1]
        last_matrix = pred_x_start[:, :, :-1, :-1]
        upstream_matrix = pred_x_start[:, :, :-1, 1:]

        # for elements which are greater than VC
        IG_l = torch.greater(last_matrix, VC)
        IG_u = torch.greater(upstream_matrix, VC)
        IG = torch.eq(IG_l, IG_u)
        IG = IG.to(input_noise.dtype)
        free_flow_loss = margin - torch.multiply(upstream_matrix - last_matrix, anchor_matrix - last_matrix)
        free_flow_loss = torch.maximum(free_flow_loss, torch.Tensor([0]).cuda())
        free_flow_loss = torch.multiply(free_flow_loss, IG)

        # for elements which are not greater than VC
        IL_l = torch.less(last_matrix, VC)
        IL_u = torch.less(upstream_matrix, VC)
        IL = torch.eq(IL_l, IL_u)
        IL = IL.to(input_noise.dtype)
        congested_loss = margin - torch.multiply(upstream_matrix - last_matrix, last_matrix - anchor_matrix)
        congested_loss = torch.maximum(congested_loss, torch.Tensor([0]).cuda())
        congested_loss = torch.multiply(congested_loss, IL)

        PC_loss = free_flow_loss + congested_loss
        PC_loss = t_decay * PC_loss
        PC_loss = torch.mean(PC_loss)

        # calculate the estimation loss
        estimation_loss = torch.square(output_noise - input_noise)
        estimation_loss = torch.multiply(weight_matrix, estimation_loss)
        estimation_loss = t_decay * estimation_loss
        estimation_loss = torch.sum(estimation_loss)
        estimation_loss = estimation_loss / train_batch_size

        x_start_loss = torch.abs(x_start - pred_x_start)
        x_start_loss = torch.multiply(x_start_loss, weight_matrix)
        x_start_loss = t_decay * x_start_loss
        x_start_loss = torch.sum(x_start_loss)
        x_start_loss = x_start_loss / train_batch_size

        # burn in period
        if current_epoch < 1000:
            custom_loss = x_start_loss
        else:
            custom_loss = x_start_loss + beta * PC_loss + gamma * Consecutive_loss

        if current_batch == 0:  # print the loss info
            print('current time stamp:', current_t)
            print('current epoch:', current_epoch, 'estimation_loss:', estimation_loss, 'spatial loss:', origin_spatial_loss,
                  'temporal loss:', origin_temporal_loss, 'PC_loss:', PC_loss)

            # real_loss = torch.mean(torch.abs(output[0, :, :, :] - gt_speed_matrix[0, :, :, :]))
            # print('Estimation loss:', estimation_loss)
        self.count += 1
        return custom_loss


def cal_mape_rmse(gt_data, sample_index, train_results_folder, TrainDataFolder, train_batch_size):
    recovered_data_folder = train_results_folder + str(sample_index) + '/'
    recovered_data = import_recovered_data(recovered_data_folder)
    recovered_data = np.array(recovered_data)
    
    train_mae = np.abs(recovered_data - gt_data)
    train_mape = train_mae/gt_data
    train_mape = np.mean(train_mape)*100
    train_rmse = np.sqrt(np.mean(np.square(train_mae)))
    train_mae = np.mean(train_mae)

    
    noisy_data_folder = TrainDataFolder
    noisy_data = import_data(noisy_data_folder)
    noisy_data = np.array(noisy_data)
    noisy_data = noisy_data[0:train_batch_size, :, :]

    origin_mae = np.abs(noisy_data - gt_data)
    origin_mape = origin_mae/gt_data
    origin_mape = np.mean(origin_mape)*100
    origin_rmse = np.sqrt(np.mean(np.square(origin_mae)))
    origin_mae = np.mean(origin_mae)
    print('Origin MAPE:', origin_mape)
    print('Origin RMSE:', origin_rmse)
    print('MAPE of the' + str(sample_index) + 'sample:', train_mape)
    print('RMSE of the' + str(sample_index) + 'sample:', train_rmse)










if __name__ == '__main__':
    # at least 2*3080Ti GPUs

    # gt data
    gt_data_folder = '../data/gt_speed_matrix/'
    gt_data = import_data(gt_data_folder)
    gt_data = np.array(gt_data)

    # import the input
    VC = 28.74
    matrix_size = 128

    # set the loss function
    max_speed = 100
    VC = (VC/max_speed)*2 - 1 # norm, used to distinguish the congested and free flow status
    # the weight of loss functions
    alpha = 1  # reconstruction loss

    beta = 10000 # phy loss 1 2
    gamma = 10000 # phy loss 3


    margin = 0.1 # hyper-pramter to avoid overfitting; 0.01,0.02,0.04,0.08,0.12
    
    count = 0  # the number of epochs, used to output results during the model training
    train_number = 500
    train_batch_size = 16
    train_number_per_epoch = np.ceil(train_number / train_batch_size)

    masked_percent_per_update = 30 # 20 - 50
    number_of_epochs_per_update = 64000000
    max_masked_percent = 50 # 20 - 50

    # Actually, we can update the weight during the model training
    save_update_weight_folder = '../data/update_weight_matrix/'


    # noise patterns
    type1 = [1]
    type2 = [1]
    for i in type1:
        for j in type2:
            loss_object = PCLoss(VC=VC, alpha=alpha, beta=beta, gamma=gamma, margin=margin, count=count, save_folder=save_update_weight_folder)
            # set the Unet
            model = df.Unet(
                dim=64,
                dim_mults=(1, 2, 4, 8)
            ).cuda()

            # set the diffusion model
            diffusion = df.GaussianDiffusion(
                model,
                loss_object,
                image_size=matrix_size,
                timesteps=1000,  # number of steps
                sampling_timesteps=1000
                # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            ).cuda()

            TrainDataFolder = '../data/demo_speed_matrix/'
            train_file_paths = obtain_file_paths(TrainDataFolder)
            print(TrainDataFolder)
            train_results_folder = '../data/results/'
            if not os.path.exists(train_results_folder):
                os.mkdir(train_results_folder)


            trainer = df.Trainer(
                diffusion,
                train_file_paths,
                '',
                max_speed=max_speed,
                train_results_folder=train_results_folder,
                test_results_folder='',
                train_batch_size=16,
                train_lr=1e-4, # learning rate
                train_num_steps=30000,  # total training steps
                gradient_accumulate_every=1,  # gradient accumulation steps
                ema_decay=0.995,  # exponential moving averamodelge decay
                save_and_sample_every=10000,
                amp=True  # turn on mixed precision
            )

            trainer.train()
            # Sample the results. The sampling is time-consuming. 
            # Theoretically, we need sample 500 speed matrices in 10 models to obtain the denosing result.
            # To save time, we just sample part of the speed matrices, that is, a batch to calculate the mape in the first three models

            for sample_index in range(1, 4):
                trainer.load(str(sample_index))
                previous_time = 1000
                trainer.train_data_remove_noise(sample_index, train_results_folder, previous_time)
                cal_mape_rmse(gt_data[0:train_batch_size,:,:], sample_index, train_results_folder, TrainDataFolder, train_batch_size)


            del trainer
            del diffusion
            del model
            del loss_object

    # print(1)
    # for i in range(2, 10):
    # trainer.recover_origin_speed_matrix(i, trainer.train_file_paths, trainer.train_results_folder)
