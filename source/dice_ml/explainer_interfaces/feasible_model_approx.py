# Dice Imports
# Pytorch
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F

import os 

from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.explainer_interfaces.feasible_base_vae import FeasibleBaseVAE
from dice_ml.utils.helpers import get_base_gen_cf_initialization

from sklearn.ensemble import RandomForestRegressor
    
class FeasibleModelApprox(FeasibleBaseVAE, ExplainerBase):

    def __init__(self, data_interface, model_interface, **kwargs):
        """
        :param data_interface: an interface class to data related params
        :param model_interface: an interface class to access trained ML model
        """

        # initiating data related parameters
        ExplainerBase.__init__(self, data_interface)

        # Black Box ML Model to be explained
        self.pred_model = model_interface.model

        self.minx, self.maxx, self.encoded_categorical_feature_indexes, \
            self.encoded_continuous_feature_indexes, self.cont_minx, self.cont_maxx, self.cont_precisions = \
            self.data_interface.get_data_params_for_gradient_dice()
        self.data_interface.one_hot_encoded_data = self.data_interface.one_hot_encode_data(self.data_interface.data_df)
        # Hyperparam
        self.encoded_size = kwargs['encoded_size']
        self.adj_matrix = kwargs['adj_matrix']
        self.learning_rate = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.validity_reg = kwargs['validity_reg']
        self.margin = kwargs['margin']
        self.epochs = kwargs['epochs']
        self.wm1 = kwargs['wm1']
        self.wm2 = kwargs['wm2']
        self.wm3 = kwargs['wm3']

        # Initializing parameters for the DiceModelApproxGenCF
        self.vae_train_dataset, self.normalise_weights, \
            self.cf_vae, self.cf_vae_optimizer = \
            get_base_gen_cf_initialization(
                self.data_interface, self.encoded_size, self.adj_matrix, self.cont_minx,
                self.cont_maxx, self.margin, self.validity_reg, self.epochs,
                self.wm1, self.wm2, self.wm3, self.learning_rate)

        # Data paths
        current_file_path = current_file_path = os.path.abspath(__file__)
        project_directory = os.path.dirname(os.path.dirname(current_file_path))
        self.base_model_dir = project_directory + '/utils/sample_trained_models/'
        self.save_path = self.base_model_dir + self.data_interface.data_name + \
            '-margin-' + str(self.margin) + '-validity_reg-' + str(self.validity_reg) + \
            '-epoch-' + str(self.epochs) + '-' + 'ae-gen' + '.pth'
        
    def train_rf_model(self, dataset, col_name_A, col_name_B):
        x = dataset[col_name_A].values
        y = dataset[col_name_B].values
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(x.reshape(-1, 1), y)
        return regr

    def compute_penalty_term_rf(self, x_pred, col_name_X, col_name_Y, rf_model, dataset):
        with torch.no_grad():
            col_idx_X = dataset.columns.get_loc(col_name_X)
            col_idx_Y = dataset.columns.get_loc(col_name_Y)
            expected_Y = rf_model.predict(x_pred[:, col_idx_X].cpu().numpy().reshape(-1, 1))
            expected_Y = torch.from_numpy(expected_Y).float().to(x_pred.device).detach()

        diff = x_pred[:, col_idx_Y] - expected_Y
        penalty_term = torch.sum(diff ** 2)
        return penalty_term

    def train(self, constraints, constraint_reg, dataset, pre_trained=False):
    
        if pre_trained:
            self.cf_vae.load_state_dict(torch.load(self.save_path))
            self.cf_vae.eval()
            return


        if constraints is None:
            constraints = []
        # Xác định trước các chỉ số cột liên quan cho mỗi tên cột trong mỗi ràng buộc
        related_col_indices_map = {}
        for constraint in constraints:
            constraint_type, constraint_variables, _ = constraint
            if constraint_type == 1:
                for col_name in constraint_variables:
                    if col_name not in related_col_indices_map:
                        related_col_indices_map[col_name] = [idx for idx, col in enumerate(dataset.columns) if col == col_name or col.startswith(col_name + "_")]
            else:
                for pair in constraint_variables:
                    col_name_A, col_name_B = pair
                    related_col_indices_map[col_name_A] = [idx for idx, col in enumerate(dataset.columns) if col == col_name_A]
                    related_col_indices_map[col_name_B] = [idx for idx, col in enumerate(dataset.columns) if col == col_name_B]

        # TODO: Handling such dataset specific constraints in a more general way
        # CF Generation for only low to high income data points
        # self.vae_train_dataset = self.vae_train_dataset[self.vae_train_dataset[:, -1] == 0, :]
        # self.vae_val_dataset = self.vae_val_dataset[self.vae_val_dataset[:, -1] == 0, :]

        # Removing the outcome variable from the datasçets
        self.vae_train_feat = self.vae_train_dataset[:, :-1]

        for epoch in range(self.epochs):
            batch_num = 0
            train_loss = 0.0
            train_size = 0

            train_dataset = torch.tensor(self.vae_train_feat).float()
            train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            for train in enumerate(train_dataset):
                self.cf_vae_optimizer.zero_grad()

                train_x = train[1]
                train_y = 1.0-torch.argmax(self.pred_model(train_x), dim=1)
                train_size += train_x.shape[0]

                out = self.cf_vae(train_x, train_y)
                
                loss = self.compute_loss(out, train_x, train_y)
                dm = out['x_pred']
                mc_samples = out['mc_samples']


                for constraint in constraints:
                    x_pred = dm[0]
                    constraint_type, constraint_variables, constraint_direction = constraint

                    constraint_loss = 0

                    if constraint_type:
                        if constraint_direction !=0:
                            for col_name in constraint_variables:
                                related_col_indices = related_col_indices_map[col_name]
                                for col_idx in related_col_indices:
                                    constraint_loss += F.hinge_embedding_loss(
                                        constraint_direction*(x_pred[:, col_idx] - train_x[:, col_idx]), torch.tensor(-1), 0)

                            for j in range(1, mc_samples):
                                x_pred = dm[j]
                                for col_name in constraint_variables:
                                    related_col_indices = related_col_indices_map[col_name]
                                    for col_idx in related_col_indices:
                                        constraint_loss += F.hinge_embedding_loss(
                                            constraint_direction*(x_pred[:, col_idx] - train_x[:, col_idx]), torch.tensor(-1), 0)
                                        
                        else:
                            for col_name in constraint_variables:
                                related_col_indices = related_col_indices_map[col_name]
                                for col_idx in related_col_indices:     
                                    difference = x_pred[:, col_idx] - train_x[:, col_idx]
                                    mask = (difference <= -0.5)
                                    filtered_difference = difference[mask]

                                    if len(filtered_difference) > 0:
                                        # Áp dụng hinge_embedding_loss
                                        constraint_loss += F.hinge_embedding_loss(filtered_difference, torch.tensor(-1), 0)/len(related_col_indices)

                                    mask = (difference >= 0.5/(len(related_col_indices)-1))
                                    filtered_difference = difference[mask]

                                    if len(filtered_difference) > 0:
                                        # Áp dụng hinge_embedding_loss
                                        constraint_loss += F.hinge_embedding_loss(filtered_difference, torch.tensor(1), 0)/len(related_col_indices)  

                            for j in range(1, mc_samples):
                                x_pred = dm[j]
                                for col_name in constraint_variables:
                                    related_col_indices = related_col_indices_map[col_name]

                                    for col_idx in related_col_indices:
                                        difference = x_pred[:, col_idx] - train_x[:, col_idx]
                                        mask = (difference <= -0.5)
                                        filtered_difference = difference[mask]

                                        if len(filtered_difference) > 0:
                                            # Áp dụng hinge_embedding_loss
                                            constraint_loss += F.hinge_embedding_loss(filtered_difference, torch.tensor(-1), 0)/len(related_col_indices)
       
                                        mask = (difference >= 0.5/(len(related_col_indices)-1))
                                        filtered_difference = difference[mask]

                                        if len(filtered_difference) > 0:
                                            # Áp dụng hinge_embedding_loss
                                            constraint_loss += F.hinge_embedding_loss(filtered_difference, torch.tensor(1), 0)/len(related_col_indices)   
                                        
                    else:
                        for pair in constraint_variables:
                            col_name_A, col_name_B = pair
                            indices_A = related_col_indices_map[col_name_A]
                            indices_B = related_col_indices_map[col_name_B]
                            diff_colA = x_pred[:, indices_A] - train_x[:, indices_A]
                            diff_colB = x_pred[:, indices_B] - train_x[:, indices_B]
                            # Tính dấu của sự khác biệt
                            sign_diff_colA = torch.sign(diff_colA)
                            sign_diff_colB = torch.sign(diff_colB)

                            # Tìm các phần tử có dấu trái nhau giữa diff_colA và diff_colB
                            opposite_sign_mask = constraint_direction * sign_diff_colA * sign_diff_colB < 0

                            # Chuyển đổi mask sang kiểu dữ liệu float để tính toán hinge loss
                            opposite_sign_float = opposite_sign_mask.float()

                            # Tính hinge loss chỉ đối với những phần tử có dấu trái nhau
                            # Sử dụng hinge loss với mục tiêu là -1 để phạt những trường hợp trái dấu
                            constraint_loss +=  F.hinge_embedding_loss(opposite_sign_float, torch.tensor(1.0), 0)
                        for j in range(1, mc_samples):
                            x_pred = dm[j]

                            for pair in constraint_variables:
                                col_name_A, col_name_B = pair
                                indices_A = related_col_indices_map[col_name_A]
                                indices_B = related_col_indices_map[col_name_B]
                                diff_colA = x_pred[:, indices_A] - train_x[:, indices_A]
                                diff_colB = x_pred[:, indices_B] - train_x[:, indices_B]
                                # Tính dấu của sự khác biệt
                                sign_diff_colA = torch.sign(diff_colA)
                                sign_diff_colB = torch.sign(diff_colB)

                                # Tìm các phần tử có dấu trái nhau giữa diff_colA và diff_colB
                                opposite_sign_mask = constraint_direction * sign_diff_colA * sign_diff_colB < 0

                                # Chuyển đổi mask sang kiểu dữ liệu float để tính toán hinge loss
                                opposite_sign_float = opposite_sign_mask.float()

                                # Tính hinge loss chỉ đối với những phần tử có dấu trái nhau
                                # Sử dụng hinge loss với mục tiêu là -1 để phạt những trường hợp trái dấu
                                constraint_loss +=  F.hinge_embedding_loss(opposite_sign_float*torch.abs(diff_colB), torch.tensor(1.0), 0)

                    constraint_loss = constraint_loss/mc_samples
                    constraint_loss = constraint_reg * constraint_loss
                    loss += constraint_loss

                loss.backward()
                train_loss += loss.item()
                self.cf_vae_optimizer.step()
                batch_num += 1

            ret = loss/batch_num

            # Save the model after training every 10 epochs and at the last epoch
            if (epoch != 0 and (epoch % 10) == 0) or epoch == self.epochs-1:
                torch.save(self.cf_vae.state_dict(), self.save_path)
