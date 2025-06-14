import json
import torch
from viz.plots import plot_voxels_batch, plot_point_cloud_batch, plot_point_cloud_batch_INR
from torchvision.utils import save_image
# from main_StegaINR import parameters_mask_generate_by_key

class TrainerINRStego():
    """Trains a function.

    Args:
        device (torch.device):
        function_distribution (models.function_distribution.FunctionDistribution):
        discriminator (models.discrimnator.PointConvDiscriminator):
        data_converter (data.conversion.{GridDataConverter, PointCloudDataConverter}):
        lr (float): Learning rate for hypernetwork.
        lr_disc (float): Learning rate for discriminator.
        betas (tuple of ints): Beta parameters to use in Adam optimizer. Usually
            this is either (0.5, 0.999) or (0., 0.9).
        r1_weight (float): Weight to use for R1 regularization.
        max_num_points (int or None): If not None, randomly samples max_num_points
            points from the coordinates and features before passing through
            generator and discriminator.
        is_voxel (bool): If True, considers data as voxels.
        is_point_cloud (bool): If True, considers data as point clouds.
        is_era5 (bool): If True, considers data as ERA5 surface temperatures.
        print_freq (int): Frequency with which to print loss.
        save_dir (string): Path to a directory where experiment logs and images
            will be saved.
        model_save_freq (int): Frequency (in epochs) at which to save model.
    """
    # modify for INR, parameter: function_representation
    def __init__(self, device, function_representation,
                 data_converter, lr=2e-4, lr_disc=2e-4, betas=(0.5, 0.999),
                 r1_weight=0, max_num_points=None, is_voxel=False,
                 is_point_cloud=False, is_era5=False, print_freq=1, save_dir='',
                 model_save_freq=0):

        self.device = device
        # Add for INR: add a function_representation class
        self.function_representation = function_representation

        self.data_converter = data_converter
        self.r1_weight = r1_weight
        self.max_num_points = max_num_points
        self.is_voxel = is_voxel
        self.is_point_cloud = is_point_cloud
        self.is_era5 = is_era5

        self.bce = torch.nn.BCELoss()

        #Add for INR: loss for INR
        self.mse = torch.nn.MSELoss()
        self.lr = lr
        self.betas = betas



        # Optimizer for function representation inr
        self.optimizer_inr = torch.optim.Adam(
            self.function_representation.forward_layers.parameters(),
            lr=lr, betas=betas
        )


        self.print_freq = print_freq
        self.save_dir = save_dir
        # Ensure number of samples saved is not too large when data itself
        # is large
        if data_converter.data_shape[1] < 65:
            self.num_samples_to_save = 32
        elif data_converter.data_shape[1] >= 65:
            self.num_samples_to_save = 16
        if self.is_voxel or self.is_point_cloud:
            self.num_samples_to_save = 16

        # Add for INR: set numsamples = 1
        self.num_samples_to_save = 1

        self.model_save_freq = model_save_freq
        self._init_logs()

    def _init_logs(self):
        """Initializes logs to track model performance during training."""
        self.logged_items = ("generator", "discriminator", "disc_real", "disc_fake")
        if self.r1_weight:
            self.logged_items = self.logged_items + ("grad_squared",)
        self.logs = {logged_item : [] for logged_item in self.logged_items}
        self.epoch_logs = {logged_item : [] for logged_item in self.logged_items}

        self.logs["inr_loss"]=list()#########


    def _log_epoch_losses(self, iterations_per_epoch):
        """
        """
        for logged_item in self.logged_items:
            self.epoch_logs[logged_item].append(mean(self.logs[logged_item][-iterations_per_epoch:]))

    def _save_logs(self):
        """
        """
        # Convert tensors in logs to Python types before saving
        serializable_logs = {}
        for logged_item, values in self.logs.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], torch.Tensor):
                # Convert list of tensors to list of Python floats
                serializable_logs[logged_item] = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
            else:
                serializable_logs[logged_item] = values

        # Save regular logs
        with open(self.save_dir + '/logs.json', 'w') as f:
            json.dump(serializable_logs, f)

        # Save epoch logs
        serializable_epoch_logs = {}
        for logged_item, values in self.epoch_logs.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], torch.Tensor):
                # Convert list of tensors to list of Python floats
                serializable_epoch_logs[logged_item] = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
            else:
                serializable_epoch_logs[logged_item] = values

        with open(self.save_dir + '/epoch_logs.json', 'w') as f:
            json.dump(serializable_epoch_logs, f)

    #add for INR: save data sample from function_representation directly
    def _save_data_samples_from_representation(self, filename):
        """
        """
        with torch.no_grad():

            samples = []
            samples.append(self.function_representation.sample_grid(self.data_converter))

            # Convert list of samples to a batch of tensors
            samples = torch.cat([sample.unsqueeze(0) for sample in samples], dim=0)
        # Save samples as grid
        if self.is_voxel:
            # Voxels lie in [0, 1], so use 0.5 as a threshold

            plot_voxels_batch(samples.detach().cpu() > .5, save_fig=self.save_dir + "/" + filename,
                              ncols=self.num_samples_to_save)# // 4
        elif self.is_point_cloud:
            plot_point_cloud_batch(samples.detach().cpu(), save_fig=self.save_dir + "/" + filename,
                              ncols=self.num_samples_to_save)# // 4)
        elif self.is_era5:
            # ERA5 data has shape (batch_size, 3, num_lats, num_lons) where 3rd
            # dimension of 2nd axis corresponds to temperature, so extract this
            # (will correspond to grayscale image)
            save_image(samples[:, 2:3].detach().cpu(), self.save_dir + "/" + filename,
                       nrow=self.num_samples_to_save // 4)
        else:
            save_image(samples.detach().cpu(), self.save_dir + "/" + filename,
                       nrow=self.num_samples_to_save // 4)

    def train(self, dataloader, epochs):
        """
        """
        self.inr_loss_min=999

        if self.save_dir:
            # add for INR: save data from function representation
            self._save_data_samples_from_representation("samples_INR_{}.png".format(0))

            # Save real samples (first few samples of dataset)
            real_samples = torch.cat([dataloader.dataset[i][0].unsqueeze(0)
                                     for i in range(self.num_samples_to_save)], dim=0)
            if self.is_voxel:
                plot_voxels_batch(real_samples.float() > .5, save_fig=self.save_dir + "/ground_truth.png",
                              ncols=self.num_samples_to_save)#// 4)
            elif self.is_point_cloud:
                real_samples = real_samples.unsqueeze(0)# from one sample to  batch samples
                plot_point_cloud_batch_INR(real_samples.float(), save_fig=self.save_dir + "/ground_truth.png",
                              ncols=self.num_samples_to_save)# // 4)
            elif self.is_era5:
                save_image(real_samples[:, 2:3], self.save_dir + "/ground_truth.png",
                           nrow=self.num_samples_to_save // 4)
            else:
                save_image(real_samples, self.save_dir + "/ground_truth_INR.png",
                           nrow=self.num_samples_to_save // 4)

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            for i, batch in enumerate(dataloader):
                self.train_batch(batch)
                if i % self.print_freq == 0:
                    if self.r1_weight:
                        print("INR_Iteration {}/{}".format(i+1,1))
                       # print("Iteration {}/{}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}, 100 x grad_squared {:.3f}".format(i + 1, len(dataloader), self.logs["generator"][-1], self.logs["discriminator"][-1], self.logs["disc_real"][-1], self.logs["disc_fake"][-1], 100 * self.logs["grad_squared"][-1]))
                    else:
                        print("INR_Iteration {}/{}".format(i + 1, 1))
                        # print("Iteration {}/{}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}".format(i + 1, len(dataloader), self.logs["generator"][-1], self.logs["discriminator"][-1], self.logs["disc_real"][-1], self.logs["disc_fake"][-1]))
            # self._log_epoch_losses(len(dataloader))##########
            if self.r1_weight:
                print("\n INR Epoch {}:".format(epoch + 1))
                # print("\nEpoch {}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}, 100 x grad_squared {:.3f}".format(epoch + 1, self.epoch_logs["generator"][-1], self.epoch_logs["discriminator"][-1], self.epoch_logs["disc_real"][-1], self.epoch_logs["disc_fake"][-1], 100 * self.epoch_logs["grad_squared"][-1]))
            else:
                print("\n INR Epoch {}:".format(epoch + 1))
                # print("\nEpoch {}: generator {:.3f}, discriminator {:.3f}, disc_real {:.3f}, disc_fake {:.3f}".format(epoch + 1, self.epoch_logs["generator"][-1], self.epoch_logs["discriminator"][-1], self.epoch_logs["disc_real"][-1], self.epoch_logs["disc_fake"][-1]))
            # Optionally save logs, samples and model
            if self.save_dir:
                self._save_logs()
                # self._save_data_samples(self.random_latents, "img_random_{}.png".format(epoch + 1))
                # Add for INR sample image from function representation
                # self._save_data_samples_from_representation("img_random_INR_stego_{}.png".format(epoch + 1))  # no need sampling during training

                # self.function_distribution.save_model(self.save_dir + "/model.pt")

                #add for INR: save: function_representation
                self.function_representation.save_model(self.save_dir + "/inr_stego_model.pt")

                # If model_save_freq is non-zero save intermediate versions of
                # the model
                if epoch != 0 and self.model_save_freq != 0 and epoch % self.model_save_freq == 0:
                    # self.function_distribution.save_model(self.save_dir + "/model_{}.pt".format(epoch))
                    self.function_representation.save_model(self.save_dir + "/inr_stego_model_{}.pt".format(epoch))

                ########
                if self.logs["inr_loss"]:
                    if min(self.logs["inr_loss"]) < self.inr_loss_min:
                        self.inr_loss_min=min(self.logs["inr_loss"])
                        self.function_representation.save_model(self.save_dir + "/inr_stego_min_loss_model.pt")
                else:pass

    def train_batch(self, batch):
        """
        """
        # Extract data
        data, _ = batch
        data = data.to(self.device)

        # Extract coordinates and features from data
        coordinates, features = self.data_converter.batch_to_coordinates_and_features(data)

        # Optionally randomly subsample coordinates and features
        if self.max_num_points:
            set_size = coordinates.shape[1]
            subset_indices = random_indices(self.max_num_points, set_size)
            # Select a subsample of coordinates and features
            coordinates = coordinates[:, subset_indices]
            features = features[:, subset_indices]

        # Add for INR: Sample  features to train function representation
        # weights, biases = self.function_representation.get_weights_and_biases
        # generated_features = self.function_representation.batch_stateless_forward_INR(coordinates, weights, biases)
        feature = torch.squeeze(features, 0)# just use one sample
        self._train_function(feature, updater=0) # use customize updater


    # Add for INR: train inr function
    def _train_function(self,features, updater):
        """
               Args:
                   features (torch.Tensor): Tensor of shape (batch_size, num_points, coordinate_dim).

               """
        #Generate features use inr model

        generated_features = self.function_representation.sample_features(self.data_converter)

        # Calculate mes losses on real and representation data
        inr_loss = self.mse(features, generated_features)
        print(f"inr_loss:{inr_loss}")#########
        self.logs["inr_loss"].append(inr_loss)#########
        if isinstance(updater, torch.optim.Optimizer):
            self.optimizer_inr.zero_grad()
            inr_loss.backward()
            self.optimizer_inr.step()
        else:
            inr_loss.sum().backward()
            # use mask for parameter update
            weights_mask, biases_mask = parameters_mask_generate_by_key("key2.json")
            batch_size = 1
            sgd_by_mask(self.function_representation.forward_layers.parameters(), self.device, self.lr, batch_size,weights_mask,biases_mask)

# Add for inr: define gradient decent algorithm with parameter mask
def sgd_by_mask(params, device, lr, batch_size, weights_mask, biases_mask):
    """
    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():

        biases_layer_index = 0
        weights_layer_index = 0
        for param in params:
            # for biases
            if len(param.shape) == 1:
                # print("before bias update: ", param)
                ones_mask = torch.ones(param.shape)
                bias_mask = biases_mask[biases_layer_index]
                biases_layer_index = biases_layer_index + 1

                sgd_mask = ones_mask - bias_mask
                # to cuda
                sgd_cuda = sgd_mask.cuda()
                param.grad *= sgd_cuda
                param -= lr * param.grad
                # print("before bias update: ", param)
            # for weights
            if len(param.shape) == 2:
                # print("before weight update: ", param)
                # ones_mask = torch.ones(param.shape)
                ones_mask = torch.ones(param.shape)
                weight_mask = weights_mask[weights_layer_index]
                weights_layer_index = weights_layer_index + 1

                sgd_mask = ones_mask-weight_mask
                # to cuda
                sgd_cuda = sgd_mask.cuda()
                # print(sgd_cuda.device)
                # print(param.grad.device)
                param.grad *= sgd_cuda
                param -=lr*param.grad
                # print("before weight update: ", param)

def norm_gradient_squared(outputs, inputs, sum_over_points=True):
    """Computes square of the norm of the gradient of outputs with respect to
    inputs.

    Args:
        outputs (torch.Tensor): Shape (batch_size, 1). Usually the output of
            discriminator on real data.
        inputs (torch.Tensor): Shape (batch_size, num_points, coordinate_dim + feature_dim)
            or shape (batch_size, num_points, feature_dim) depending on whether gradient
            is over coordinates and features or only features.
        sum_over_points (bool): If True, sums over num_points dimension, otherwise takes mean.

    Notes:
        This is inspired by the function in this repo
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    """
    batch_size, num_points, _ = inputs.shape
    # Compute gradient of outputs with respect to inputs
    grad_outputs = torch.autograd.grad(
        outputs=outputs.sum(), inputs=inputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    # Square gradients
    grad_outputs_squared = grad_outputs.pow(2)
    # Return norm of squared gradients for each example in batch. We sum over
    # features, to return a tensor of shape (batch_size, num_points).
    regularization = grad_outputs_squared.sum(dim=2)
    # We can now either take mean or sum over num_points dimension
    if sum_over_points:
        return regularization.sum(dim=1)
    else:
        return regularization.mean(dim=1)


def random_indices(num_indices, max_idx):
    """Generates a set of num_indices random indices (without replacement)
    between 0 and max_idx - 1.

    Args:
        num_indices (int): Number of indices to include.
        max_idx (int): Maximum index.
    """
    # It is wasteful to compute the entire permutation, but it looks like
    # PyTorch does not have other functions to do this
    permutation = torch.randperm(max_idx)
    # Select first num_indices indices (this will be random since permutation is
    # random)
    return permutation[:num_indices]

def mean(array):
    """Returns mean of a list.

    Args:
        array (list of ints or floats):
    """
    return sum(array) / len(array)

#Add for INR
def parameters_mask_generate_by_key(key_json_file):
    with open(key_json_file) as f:
        key = json.load(f)

    # biases: just remove input layer
    biases_mask = key.copy()
    biases_mask.pop(0)   #remove input layer

    # convert list to tensor list
    biases_mask_tensor_list = []
    for item in biases_mask:
        biases_mask_tensor_list.append(torch.Tensor(item))

    weights_mask_tensor_list=[]
    for layer, pre_layer in zip(biases_mask, key):
        m = len(layer)
        n = len(pre_layer)
        weight_mask = torch.zeros([m, n])
        for i in range(m):
            for j in range(n):
                if layer[i]==1 and pre_layer[j]== 1:
                    weight_mask[i, j] = 1
        weights_mask_tensor_list.append(weight_mask)

    return weights_mask_tensor_list, biases_mask_tensor_list