from copy import deepcopy

import models, torch
import numpy as np


class Server(object):

    # 加载全局模型
    def __init__(self, conf, eval_dataset):

        # 最小生成树结构
        self.part_connect_graph = []

        # 所有的客户端的secretkey和bu的份额
        self.all_part_secretkey_bu = {}

        self.client_dict = {}
        self.client_list = []

        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def reveive_msg(self):
        pass

    # 收集来自客户端的份额
    def collect_shared_secretkey_bu(self, client_shared_key_bu):
        origin_client_id = list(client_shared_key_bu.keys())[0]
        for client_id in client_shared_key_bu[origin_client_id]:
            if client_id in self.all_part_secretkey_bu.keys():
                self.all_part_secretkey_bu[client_id].append(
                    {origin_client_id: client_shared_key_bu[origin_client_id][client_id]})
            else:
                self.all_part_secretkey_bu[client_id] = [
                    {origin_client_id: client_shared_key_bu[origin_client_id][client_id]}]

    # 根据份额重建密钥和bu
    def reconstruct_secretkey_bu(self, t, client_shared_key_bu):
        for i in range(len(client_shared_key_bu)):
            for j in range(i + 1, len(client_shared_key_bu)):
                if list(client_shared_key_bu[j].keys())[0] < list(client_shared_key_bu[i].keys())[0]:
                    temp = client_shared_key_bu[i]
                    client_shared_key_bu[i] = client_shared_key_bu[j]
                    client_shared_key_bu[j] = temp

        if len(client_shared_key_bu) >= t:
            y_bu = []
            y_secretkey = []
            id_list = []
            for item in client_shared_key_bu:
                _id = list(item.keys())[0]
                id_list.append(_id)
                # [secretkey, bu]
                y_secretkey.append(item[_id][0])
                y_bu.append(item[_id][1])
            A = []
            for i in range(t):
                a = [1]
                for j in range(t - 1):
                    # a.append((i + 1) ** (j + 1))
                    a.append(int(id_list[i]) ** (j + 1))
                A.append(a)
            A_bu = []
            A_secretkey = []
            for i in range(t):
                a_bu = [y_bu[i]]
                a_secretkey = [y_secretkey[i]]
                for j in range(t - 1):
                    a_bu.append(A[i][j + 1])
                    a_secretkey.append(A[i][j + 1])
                A_bu.append(a_bu)
                A_secretkey.append(a_secretkey)
            bu = np.linalg.det(A_bu) / np.linalg.det(A)
            k = np.linalg.det(A_secretkey) / np.linalg.det(A)
            # 取整
            if k - int(k) > 0.9:
                k = int(k) + 1
            else:
                k = int(k)
            if bu - int(bu) > 0.9:
                bu = int(bu) + 1
            else:
                bu = int(bu)
            return [k, bu]
        else:
            pass
        return []

    # 生成噪声张量（加噪声） PRG伪随机生成器，seed一样，随机向量也一样
    # def generate_weights(self, seed, dim, _type):
    #     # 定义随机数种子
    #     np.random.seed(seed)
    #     # 生成dim维度的向量
    #     if len(dim) == 0:
    #         if _type == np.int64:
    #             return np.int64(np.random.rand())
    #         if _type == np.float32:
    #             return np.int64(np.random.rand())
    #     if len(dim) == 1:
    #         if _type == np.int64:
    #             return np.int64(np.random.rand(dim[0]))
    #         if _type == np.float32:
    #             return np.float32(np.random.rand(dim[0]))
    #     if len(dim) == 2:
    #         if _type == np.int64:
    #             return np.int64(np.random.rand(dim[0], dim[1]))
    #         if _type == np.float32:
    #             return np.float32(np.random.rand(dim[0], dim[1]))
    #     if len(dim) == 3:
    #         if _type == np.int64:
    #             return np.int64(np.random.rand(dim[0], dim[1], dim[2]))
    #         if _type == np.float32:
    #             return np.float32(np.random.rand(dim[0], dim[1], dim[2]))
    #     if len(dim) == 4:
    #         if _type == np.int64:
    #             return np.int64(np.random.rand(dim[0], dim[1], dim[2], dim[3]))
    #         if _type == np.float32:
    #             return np.float32(np.random.rand(dim[0], dim[1], dim[2], dim[3]))

    def generate_weights(self, seed, dim, _type):
        torch.manual_seed(seed)
        if len(dim) == 0:
            return torch.tensor(0)
        elif len(dim) == 1:
            if _type == torch.int64:
                return torch.randn(dim[0])
            elif _type == torch.float32:
                return torch.randn(dim[0], dtype=torch.float32)
        elif len(dim) == 2:
            if _type == torch.int64:
                return torch.randn((dim[0], dim[1]))
            elif _type == torch.float32:
                return torch.randn((dim[0], dim[1]), dtype=torch.float32)
        elif len(dim) == 3:
            if _type == torch.int64:
                return torch.randn((dim[0], dim[1], dim[2]))
            elif _type == torch.float32:
                return torch.randn((dim[0], dim[1], dim[2]), dtype=torch.float32)
        elif len(dim) == 4:
            if _type == torch.int64:
                return torch.randn((dim[0], dim[1], dim[2], dim[3]))
            elif _type == torch.float32:
                return torch.randn((dim[0], dim[1], dim[2], dim[3]), dtype=torch.float32)

    # 服务器如果没有收到某个客户端的梯度，就会自己生成掩码去unmask
    def reveal(self, keylist):
        wghts = np.zeros(self.dim)
        for each in keylist:
            if each < self.id:
                wghts -= self.generate_weights((self.keys[each] ** self.secretkey) % 17)
            elif each > self.id:
                wghts += self.generate_weights((self.keys[each] ** self.secretkey) % 17)
        return -1 * wghts

    def unmask(self):
        for client_id in self.all_part_secretkey_bu:
            # 重构key bu
            secretkey_bu = self.reconstruct_secretkey_bu(self.conf["t"], self.all_part_secretkey_bu[client_id])
            # 消除bu掩码
            for name, data in self.global_model.state_dict().items():
                _data = data
                dim = _data.shape
                _type = _data.dtype
                bu_mask = -self.conf["lambda"] * self.generate_weights(secretkey_bu[1], dim, _type)
                if data.type() != bu_mask.type():
                    data.add_(bu_mask.to(torch.int64))
                else:
                    data.add_(bu_mask)

    # 模型聚合函数agg
    # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一层乘上学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 累加和
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 模型评估
    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            # sum up batch loss
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
