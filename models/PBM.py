import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from models.resnet_features import resnet50_features
from torchattacks import PGD, PGD_V2V
from utils import AverageMeter

class PBM(nn.Module):

    def __init__(self, conf):
        super(PBM, self).__init__()
        backbone = conf["model_args"]["base"]
        prototype_shape = (2000, 128, 1, 1)
        self.img_size = 224
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = conf["model_args"]["num_classes"]

        self.epsilon = 1e-4
        
        if backbone == 'resnet50':
            self.backbone = resnet50_features(conf["model_args"]["use_pretrained"])
        
        self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes).cuda()

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias
        
        self._initialize_weights()
        self.atk_mode = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), **conf["optimizer_args"]
        )
        self.fc_optimizer = torch.optim.SGD(
            self.last_layer.parameters(), **conf["fc_optimizer_args"]
        )
        self.use_adv = conf["use_adv"]
        self.use_noise = conf["use_noise"]
        self.conf = conf

    def forward(self, x):
        # extract features
        conv_features = self.backbone(x)
        conv_features = self.add_on_layers(conv_features)
        
        # use (x-a)^2 to cal distances
        distances = self._l2_convolution(conv_features)
        
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)

        if self.atk_mode == True:
            return logits
        else:
            return min_distances, logits

    def Joint(self, loader):
        model = self
        optimizer = self.optimizer
        fc_optimizer = self.fc_optimizer
        label_loss_meter = AverageMeter()
        label_acc_meter = AverageMeter()
        for data in loader:
            if len(data) == 2:
                img, label = data
            else:
                img, label, _ = data
            
            if "image2label" in self.use_adv:
                self.atk_mode = True
                atk = PGD(self, eps=5 / 255, alpha=2 / 225, steps=2, random_start=True)
                atk.set_normalization_used(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                adv_img = atk(img, label).cpu()
                adv_label = label.clone().detach().cpu()
                img = torch.cat([img, adv_img], dim=0)
                label = torch.cat([label, adv_label], dim=0)
                self.atk_mode = False
            img, label = img.cuda(), label.cuda()
            min_distances, label_pred = model(img)
            # cal label loss
            label_loss = self.loss_fn(label_pred, label)

            # cal interpretation loss
            max_dist = (
                model.prototype_shape[1]
                * model.prototype_shape[2]
                * model.prototype_shape[3]
            )

            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(
                model.prototype_class_identity[:, label]
            )
            inverted_distances, _ = torch.max(
                (max_dist - min_distances) * prototypes_of_correct_class, dim=1
            )
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = torch.max(
                (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
            )
            separation_cost = torch.mean(
                max_dist - inverted_distances_to_nontarget_prototypes
            )

            # calculate avg cluster cost
            avg_separation_cost = torch.sum(
                min_distances * prototypes_of_wrong_class, dim=1
            ) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            l1_mask = 1 - torch.t(model.prototype_class_identity)
            l1 = (model.last_layer.weight * l1_mask).norm(p=1)

            # cluster_losses.append(cluster_cost.item())
            # separation_losses.append(separation_cost.item())
            # l1_losses.append(l1.item())

            # backward
            loss = label_loss + 1.0 * cluster_cost - 0.1 * separation_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            label_pred = torch.argmax(label_pred, dim=1)
            correct = torch.sum(label_pred == label).int().sum().item()
            num = len(label)
            label_loss_meter.update(label_loss.item(), num)
            label_acc_meter.update(correct / num, num)

        return {
            "label_loss": label_loss_meter.avg,
            "label_acc": label_acc_meter.avg,
        }

    def run_epoch(self, loader):
        if self.conf["trainer"] == "Joint":
            return self.Joint(loader)

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)
        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)
        
        return distances
        
    def distance_2_similarity(self, distances):
        # if self.prototype_activation_function == 'log':
        return torch.log((distances + 1) / (distances + self.epsilon))
        # elif self.prototype_activation_function == 'linear':
        #     return -distances
        # else:
        #     return self.prototype_activation_function(distances)
        
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

