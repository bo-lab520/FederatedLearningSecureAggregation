import models, torch
import numpy as np


class Server(object):

    # 加载全局模型
    def __init__(self, conf, eval_dataset):

        # 最小生成树结构
        self.part_connect_graph = []

        # 所有的客户端的secretkey和bu的份额
        self.all_part_secretkey_bu = {}

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
            y = []
            for item in client_shared_key_bu:
                _id = list(item.keys())[0]
                y.append(item[id])
            A = []
            for i in range(t):
                a = [1]
                for j in range(t - 1):
                    # a.append(((i + 1) ** (j + 1)) % mod)
                    a.append((i + 1) ** (j + 1))
                A.append(a)
            _A = []
            for i in range(t):
                a = [y[i]]
                for j in range(t - 1):
                    a.append(A[i][j + 1])
                _A.append(a)
            k = np.linalg.det(_A) / np.linalg.det(A)
            return k
        else:
            pass
        return -1

    def unmask(self):
        weight_accumulator = {}
        for name, params in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for client_id in self.all_part_secretkey_bu:
            bu = self.reconstruct_secretkey_bu(3, self.all_part_secretkey_bu[client_id])
            np.random.seed(bu)
            for name, params in self.global_model.state_dict().items():
                item = params.detach().numpy()
                dim = item.shape
                bu_mask = np.float32(np.random.rand(dim))
                weight_accumulator[name].add_(-1 * bu_mask)

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
                # print(name)
                # print(data)
                data.add_(update_per_layer)
                # print(data)

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
