import asyncio
import copy
import os
import os.path as osp
from collections import defaultdict
from enum import Enum
from logging import getLogger
from typing import List, Any
import copy

import numpy as np
import torch
from pydantic import BaseModel
from rapidfuzz import process, fuzz
from scipy import spatial
from torch import nn
from tqdm import tqdm

from sklearn.cluster import KMeans
import numpy as np


from agentverse.initialization import load_agent, load_llm
from model.RecNet_base import RecNet
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

from FlagEmbedding import FlagModel


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)


class NodeType(Enum):
    user = 'user'
    item = 'item'
    attribute = 'attribute'


class Node(BaseModel, frozen=True):
    type: NodeType
    value: Any

class Node_cluster:
    def __init__(self, description, attr_list, update_attr_list):
        self.description = description
        self.attr_list = attr_list
        self.update_attr_list = update_attr_list
        # description: str = ""
        # attr_list: List = []
        # update_attr_list: List = []

class RecNet_Router(RecNet):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        SequentialRecommender.__init__(self, config, dataset)

        self.dataset_name = dataset.dataset_name
        self.n_users = dataset.num(self.USER_ID)
        self.item_token_id = dataset.field2token_id['item_id']
        self.item_id_token = dataset.field2id_token['item_id']
        self.user_id_token = dataset.field2id_token['user_id']
        self.user_token_id = dataset.field2token_id['user_id']

        self.config = config
        self.data_path = config['data_path']
        self.max_his_len = config['max_his_len']

        self.model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)

        self.logger = getLogger()
        self.record_idx = 0
        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}', )
            if os.path.exists(path):
                self.record_idx += 1
                continue
            else:
                break
        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")

        embed_model = load_llm({
            'llm_type': 'openai-embedding',
            'model_name': self.config['embed_model_name'],
            'model_list_file_path': self.config['model_list_file_path']
        })
        embedding_context = {
            'agent_type': 'embeddingagent',
            'llm': embed_model,
            'prompt_template': '',
            'output_parser_type': 'recommender',
        }
        self.embedding_agent = load_agent(embedding_context)

        llm = load_llm({
            'llm_type': 'openai-chat',
            'model_name': self.config['chat_model_name'],
            'model_list_file_path': self.config['model_list_file_path'],
            'sampling_params': self.config['sampling_params']
        })
        update_llm = load_llm({
            'llm_type': 'openai-chat',
            'model_name': self.config['reason_model_name'],
            'model_list_file_path': self.config['model_list_file_path'],
            'sampling_params': self.config['reason_sampling_params']
        })

        user2context = self._load_user_context(update_llm)

        # graph_cluster新增：
        # attribute_all存当前所有的属性词语的字典，key：attr，value：number（数量）
        self.attribute_all = []

        self.kmeans_k = config["kmeans_count"]
        self.cluster_all = [] # 用来存Node_cluster

        self.update_cluster_id = [] # 存储每次更新的cluster id，从而在propagate的时候进行

        self.current_user_id = [] # 记录每次成功加入的user数据

        # 结构体列表这边写错了，这里约等于没有建立实例，如果是这个类里比较
        # 确定计算边的分数，这一点也很重要
        for i in range(self.kmeans_k):
            self.cluster_all.append(Node_cluster(description="",attr_list=[], update_attr_list=[]))
        
        # self.cluster_all[0].attr_list.append("hi")

        # for i in range(self.kmeans_k):
        #     print(self.cluster_all[i].attr_list)

        self.user_agents = {}
        for user_id, user_context in user2context.items():
            agent = load_agent(user_context) # 初始化对应的partition信息
            self.user_agents[user_id] = agent
            user_id = str(user_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}', )
            user_description = user_context['memory_1'][-1]
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{user_id}'), 'w') as f:
                f.write('~' * 20 + 'Meta information' + '~' * 20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_description}\n')

        self.item_text = self._load_text()

        item2context = self._load_item_context(update_llm)

        self.item_agents = {}
        for item_id, item_context in item2context.items():
            agent = load_agent(item_context)
            self.item_agents[item_id] = agent
            item_id = str(item_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}', )
            item_description = item_context['role_description']
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'item.{item_id}'), 'w') as f:
                f.write('~' * 20 + 'Meta information' + '~' * 20 + '\n')
                f.write(f'The item has the following characteristics: {item_description} \n')

        rec_context = {
            'agent_type': 'recagent',
            'memory': [],

            'llm': llm,

            'prompt_template': self.config['system_prompt_template'],
            'system_prompt_template_backward': self.config['system_prompt_template_backward'],
            'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],
            'system_prompt_template_evaluation_sequential': self.config[
                'system_prompt_template_evaluation_sequential'],
            'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],

            'output_parser_type': 'recommender',
        }
        self.rec_agent = load_agent(rec_context)

        self.dummy_parameter = nn.Parameter(torch.Tensor())

        self.edge_dict = defaultdict(lambda: defaultdict(set))
        self.uu_dict = defaultdict(set)
        self.ii_dict = defaultdict(set)
        self.batch_attribute_count = dict() # 记录全局attribute记录的节点个数

        self.user_dict = dict()
        self.item_dict = dict()
        self.attr_dict = dict()

        self.user_attr_set_dict = dict() # 记录每个user的attr列表，用来比较相似度

        self.graph_agent = load_agent({
            'agent_type': 'graph_agent',
            'llm': llm,
        })

        self.propagate_agent = load_agent({
            'agent_type': 'propagate_agent',
            'llm': update_llm,
        })

    def _load_user_context(self, llm):
        user2context = {}
        for user_id in range(self.n_users):
            if user_id == 0:
                init_memory = '[PAD]'
                init_group_memory = '[PAD]'
            else:
                init_memory = ' I enjoy listening to CDs very much.'
                init_group_memory = 'We enjoy listening to CDs very much.'

            user2context[user_id] = {
                'agent_type': 'useragent',

                'memory_1': [init_memory],
                'update_memory': [init_memory],
                'group_memory_1': [init_group_memory],
                'update_group_memory': [init_group_memory],

                # 所有的当前属性集合
                'attribute_set': [], # 当前user所具有的属性列表

                'historical_interactions': [],

                'llm': llm,

                'user_prompt_system_role': self.config['user_prompt_system_role'],
                'prompt_template': self.config['user_prompt_template'],
                'user_prompt_template_true': self.config['user_prompt_template_true'],

                'output_parser_type': 'useragent',
            }
        return user2context

    def _load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'CDs.item')
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                try:
                    item_id, movie_title, genre = line.strip().split('\t')
                except ValueError:
                    print(line)
                    item_id, movie_title = line.strip().split('\t')
                token_text[item_id] = movie_title
        for token in self.item_id_token:
            if token == '[PAD]':
                continue
            item_text.append(token_text[token])
        return item_text

    def _load_item_context(self, llm):
        item_context = {0: {
            'agent_type': 'itemagent',

            'role_description': {'item_title': '[PAD]', 'item_release_year': '[PAD]',
                                 'item_class': '[PAD]'},
            # 'memory': ['[PAD]'],
            'update_memory': ['[PAD]'],

            'llm': llm,

            'prompt_template': self.config['user_prompt_template'],
            'item_prompt_template_true': self.config['item_prompt_template_true'],

            'output_parser_type': 'itemagent'
        }}
        init_item_descriptions = []
        feat_path = osp.join(self.data_path, f'CDs.item')
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                try:
                    item_id, item_title, item_class = line.strip().split('\t')
                except ValueError:
                    item_id, item_title = line.strip().split('\t')
                    item_class = 'CDs'
                if item_id not in self.item_token_id:
                    continue
                role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'. The category of this CD is: '{item_class}'."
                # role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'."
                item_context[self.item_token_id[item_id]] = {
                    'agent_type': 'itemagent',

                    'role_description': {'item_title': self.item_text[self.item_token_id[item_id]],
                                         'item_class': item_class},
                    'update_memory': [role_description_string],

                    'attribute_set' : [],

                    'llm': llm,

                    'prompt_template': self.config['item_prompt_template'],
                    'item_prompt_template_true': self.config['item_prompt_template_true'],

                    'output_parser_type': 'itemagent'
                }
                init_item_descriptions.append(role_description_string)

        for i, item in enumerate(item_context.keys()):
            if item == 0:
                continue
            item_context[item]['memory_embedding'] = {init_item_descriptions[i - 1]: None}

        return item_context
    
    async def _update_graph_cluster(self, update_user_set, update_item_set):
        print("hello hello")
        update_user_list = list(update_user_set)
        update_item_list = list(update_item_set)

        # 先统计出当前所有的attribute list，这个非常重要
        # 1. 获取所有更新user和item的profile然后提取属性放在agent结构中
        updated_user_description_list = [self.user_agents[int(user_id)].update_memory[-1] for user_id in update_user_list]
        task_list = [self.graph_agent.get_user_attr(updated_user_description) for updated_user_description in updated_user_description_list]
        attributes_list_user = await asyncio.gather(*task_list)

        updated_item_description_list = [self.item_agents[int(item_id)].update_memory[-1] for item_id in update_item_list]
        task_list = [self.graph_agent.get_item_attr(updated_item_description) for updated_item_description in updated_item_description_list]
        attributes_list_item = await asyncio.gather(*task_list)

        # 2. 遍历所有的attribute然后进行组合

        # 所有没出现在all中的就是还没有被聚类的，新的都是需要被聚类的
        non_cluster_attr_set = set() # 这些都是新被添加进入all attribute list的属性

        # 每个更新user和item都需要维护一个属于自己的属性列表
        for user_id, attributes in zip(update_user_list, attributes_list_user):
            if attributes == "":
                continue

            for attr in attributes:
                if attr not in self.attribute_all and len(self.attribute_all) > 0:
                    sim_attr, score = process.extractOne(attr, self.attribute_all, scorer=fuzz.QRatio)[:2]
                    if score > 90:
                        attr = sim_attr

                if attr not in self.attribute_all:
                    self.attribute_all.append(attr) # 保证了全体attr属性的唯一性
                    non_cluster_attr_set.add(attr)

                # 更新到user的属性列表里
                if attr not in self.user_agents[int(user_id)].attribute_set:
                    self.user_agents[int(user_id)].attribute_set.append(attr)

        for item_id, attributes in zip(update_item_list, attributes_list_item):
            if attributes == "":
                continue

            for attr in attributes:
                if attr not in self.attribute_all and len(self.attribute_all) > 0:
                    sim_attr, score = process.extractOne(attr, self.attribute_all, scorer=fuzz.QRatio)[:2]
                    if score > 90:
                        attr = sim_attr

                if attr not in self.attribute_all:
                    self.attribute_all.append(attr) # 这里是保证唯一的，其他的都不重要
                    non_cluster_attr_set.add(attr)

                # 更新到user的属性列表里
                if attr not in self.item_agents[int(item_id)].attribute_set: # 这里的set都是list，所以直接使用append
                    self.item_agents[int(item_id)].attribute_set.append(attr)


        # 3. 合并none_cluster_list和self.cluster_all
        non_cluster_attr_list = list(non_cluster_attr_set)

        all_cluster_text = []  # 存储包括聚类文本在内的所有待kmeans文本
        is_cluster = 0 # 指标，0的时候说明是原始的，1的时候说明已经有kmeans_k个聚类文本加进来了

        for node_cluster in self.cluster_all:
            if node_cluster.description != "":
                is_cluster = 1
                all_cluster_text.append(node_cluster.description)

        for attr in non_cluster_attr_list:
            all_cluster_text.append(attr)

        # 开始embedding作聚类
        embeddings = self.model.encode(all_cluster_text) # 获取对应的embeddings
        # embeddings = embeddings.cpu().numpy()

        # 开始Kmeans
        if is_cluster == 0: # 不直接初始化
            kmeans = KMeans(n_clusters=self.kmeans_k, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        elif is_cluster == 1:
            select_indices = [x for x in range(self.kmeans_k)]
            init_centroid = embeddings[select_indices]
            kmeans = KMeans(n_clusters=self.kmeans_k, init=init_centroid, n_init=1)
            clusters = kmeans.fit_predict(embeddings)

        update_cluster_id = set()

        if is_cluster == 0: 
            for text, cluster in zip(all_cluster_text, clusters):
                self.cluster_all[cluster].attr_list.append(text)
                self.cluster_all[cluster].update_attr_list.append(text)
                update_cluster_id.add(cluster)
        elif is_cluster == 1: 
            for text, cluster in zip(all_cluster_text[self.kmeans_k:], clusters[self.kmeans_k:]):
                self.cluster_all[cluster].attr_list.append(text)
                self.cluster_all[cluster].update_attr_list.append(text)
                update_cluster_id.add(cluster)

        updated_cluster_descriptions = []

        for cluster_index in range(self.kmeans_k):
            print(cluster_index)
            print(self.cluster_all[cluster_index].attr_list)
        
        self.update_cluster_id = list(update_cluster_id) 

        for update_cluster in update_cluster_id:
            if self.cluster_all[update_cluster].description != "":
                update_str = self.cluster_all[update_cluster].description + "," 
            else:
                update_str = self.cluster_all[update_cluster].description

            attr_str = ",".join(self.cluster_all[update_cluster].update_attr_list)
            total_str = update_str + attr_str

            self.cluster_all[update_cluster].update_attr_list = []

            updated_cluster_descriptions.append(total_str)


        task_list = [self.graph_agent.get_cluster_description(updated_cluster_description) for updated_cluster_description in updated_cluster_descriptions]
        new_cluster_description = await asyncio.gather(*task_list)

        index = 0
        for update_cluster in update_cluster_id:
            if new_cluster_description[index] != "":
                self.cluster_all[update_cluster].description = new_cluster_description[index]
            index += 1

    async def _propagate_update_cluster(self, update_user_set):
        task_list = []
        user_id_list = []
        for index in self.current_user_id:
            user_description = self.user_agents[index].update_memory[-1]

            original_group_description = self.user_agents[index].update_group_memory[-1]

            # 计算排序就行了
            score = []
            count_0 = 0

            # print("user list")
            # print(self.user_agents[index].attribute_set)
            for cluster_index in range(self.kmeans_k):
                count = [word for word in self.user_agents[index].attribute_set if word in self.cluster_all[cluster_index].attr_list]
                # print(self.cluster_all[cluster_index].attr_list) # cluster_index在代码里没有变过
                total_len = len(self.user_agents[index].attribute_set) + len(self.cluster_all[cluster_index].attr_list) - len(count)

                score.append(len(count) / total_len)
                if len(count) == 0:
                    count_0 += 1

            print(self.kmeans_k - count_0) # emmmm 几乎没有不是0的，这太可怕了

            # 暂时选取top 2:
            sorted_indices = [
                index for index, value in sorted(
                    enumerate(score),
                    key=lambda x: x[1],  # 按值排序
                    reverse=True         # 降序
                )
            ]

            print([score[x] for x in sorted_indices]) # 从高到低排列分数
            sorted_final = sorted_indices[:2] # 可以变成动态选择的性质，在其基础上取topk个就行了

            # 看是否需要更新
            adding = 0
            for indexing in sorted_final:
                if indexing in self.update_cluster_id:
                    adding = 1


            if adding == 1:
                update_nb_user_description_list = [self.cluster_all[cluster_id] for cluster_id in sorted_indices[:2]]
                task_list.append(self.propagate_agent.update_description(user_description, update_nb_user_description_list, original_group_description))
                user_id_list.append(index) # 添加当前user

        # 将task list分开
        updated_group_description_list = await asyncio.gather(*task_list)
        self.logger.info(f'update number {len(updated_group_description_list)}')

        for user_id, updated_group_description in zip(user_id_list, updated_group_description_list):
            if updated_group_description != "":
                self.user_agents[user_id].update_group_memory.append(updated_group_description) 
 

    async def _propagate_update(self, update_user_set):
        # 要分开来处理
        update_user_node_set = set(self.user_dict[user_id] for user_id in update_user_set)
        # # propagate may update the user in the set again, so we need to store their memory before propagation
        # update_user_description_dict = {user_node: self.user_agents[user_node.value].update_memory[-1] for user_node in update_user_node_set}

        task_list = []
        user_id_list = []
        ori_user_description_list = []

        nb_user_ids_list = []
        nb_user_descriptions_list = []

        for user_node in self.user_dict.values():
            update_nb_user_node_set = update_user_node_set & self.uu_dict[user_node] 
            if user_node in update_nb_user_node_set:
                update_nb_user_node_set.remove(user_node) 

            if len(update_nb_user_node_set) > 0:
                user_id = user_node.value
                user_id_list.append(user_id)

                user_description = self.user_agents[user_id].update_memory[-1]

                original_group_description = self.user_agents[user_id].update_group_memory[-1]

                update_nb_user_description_list = [self.user_agents[user_node.value].update_memory[-1] for user_node in update_nb_user_node_set]

                task_list.append(self.propagate_agent.update_description(user_description, update_nb_user_description_list, original_group_description))

                ori_user_description_list.append(user_description)

                nb_user_ids_list.append(update_nb_user_node_set) 
                nb_user_descriptions_list.append(update_nb_user_description_list)

        updated_group_description_list = await asyncio.gather(*task_list)
        self.logger.info(f'update number {len(updated_group_description_list)}')

        for user_id, updated_group_description, ori_user_description, nb_user_ids, nb_user_descriptions in zip(user_id_list, updated_group_description_list, ori_user_description_list, nb_user_ids_list, nb_user_descriptions_list):
            if updated_group_description != "":
                self.user_agents[user_id].update_group_memory.append(updated_group_description) 

            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            os.makedirs(path, exist_ok=True)
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'Update group preference by propagation' + '~' * 20 + '\n')
                f.write(f'neighbour users: {nb_user_ids}\n\n')

                nb_user_descriptions_str = '\n===\n'.join(nb_user_descriptions)
                f.write(f'neighbour user descriptions:\n{nb_user_descriptions_str}\n\n')

                f.write(f'original group preference description: {original_group_description}\n\n')
                f.write(f'propagated group preference description: {updated_group_description}\n\n')

    async def calculate_loss(self, interaction):
        print(f"User ID is : {interaction[self.USER_ID]}")
        print(f"Item ID is : {interaction[self.ITEM_ID]}")

        batch_user = interaction[self.USER_ID]
        batch_size = batch_user.size(0)

        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]

        batch_item_list = interaction[self.ITEM_SEQ]

        update_user_set = set()

        update_item_set = set()

        first_time = set()

        for i in range(self.config['all_update_rounds']):
            print("~" * 20 + f"{i}-th round update!" + "~" * 20 + '\n')

            # TODO: forward part i.e. candidate item selection
            user_forward_description, pos_item_forward_description, neg_item_forward_description = [], [], []
            for j in range(batch_size):
                if int(batch_user[j]) not in self.current_user_id:
                    self.current_user_id.append(int(batch_user[j]))
                user_forward_description.append(self.user_agents[int(batch_user[j])].update_memory[-1])
                pos_item_forward_description.append(self.item_agents[int(batch_pos_item[j])].update_memory[-1])
                neg_item_forward_description.append(self.item_agents[int(batch_neg_item[j])].update_memory[-1])

            system_selections, system_reasons = await self.forward(batch_user, batch_pos_item, batch_neg_item)

            accuracy_list = self._convert_system_selections_to_accuracy(system_selections, batch_pos_item,
                                                                        batch_neg_item)
            print(f"Current accuracy is {np.mean(accuracy_list)}")

            backward_system_reasons, backward_user, backward_pos_item, backward_neg_item, backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true = [], [], [], [], [], [], [], []
            
            for j, acc in enumerate(accuracy_list):
                if acc == 0:  # record wrong choices
                    backward_pos_item.append(int(batch_pos_item[j]))
                    backward_neg_item.append(int(batch_neg_item[j]))
                    backward_user.append(int(batch_user[j]))
                    update_user_set.add(int(batch_user[j]))
                    update_item_set.add(int(batch_pos_item[j]))
                    update_item_set.add(int(batch_neg_item[j]))
                    backward_system_reasons.append(system_reasons[j])
                    # 都是添加对应的元素，这样可以多尝试一下
                else:  # record right choices
                    if i == 0:
                        first_time.add(int(batch_user[j]))
                        backward_user_true.append(int(batch_user[j]))
                        update_user_set.add(int(batch_user[j]))
                        update_item_set.add(int(batch_pos_item[j]))
                        update_item_set.add(int(batch_neg_item[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])
                    elif int(batch_user[j]) not in first_time:
                        backward_user_true.append(int(batch_user[j]))
                        update_user_set.add(int(batch_user[j]))
                        update_item_set.add(int(batch_pos_item[j]))
                        update_item_set.add(int(batch_neg_item[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])

            # if sum(accuracy) / len(accuracy) > 0.9: break
            print(f"the user to update: {backward_user}")
            await self._backward(backward_system_reasons, backward_user, backward_pos_item, backward_neg_item)
            if i == 0 and len(backward_user_true) > 0:
                await self._backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true,
                                          backward_neg_item_true, round_1=True)

        await self._backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true,
                                  backward_neg_item_true, round_1=False)

        # system_reasons_embeddings = [None for _ in range(len(system_reasons))]
        # if self.config['evaluation'] == 'rag':
        #     system_reasons_embeddings = self._generate_embedding(system_reasons)
        #
        # for i, user in enumerate(batch_user):
        #     self.rec_agent.user_examples[int(user)][(
        #         user_forward_description[i], self.item_text[int(batch_pos_item[i])],
        #         self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i],
        #         neg_item_forward_description[i], accuracy_list[i], system_reasons[i])] = system_reasons_embeddings[i]

        # await self._update_graph_element(update_user_set) # 针对所有更新的user，拿其他的user来进行更新
        # await self._propagate_update(update_user_set)

        await self._update_graph_cluster(update_user_set, update_item_set)
        await self._propagate_update_cluster(update_user_set)

        # self.logger.info(f'current update relation: {dict(self.edge_dict)}')

        self._logging_after_updation(batch_user, batch_pos_item, batch_neg_item)

        batch_pos_item_descriptions = []
        batch_neg_item_descriptions = []
        for i in range(batch_size): # 这些都是更新过的
            self.user_agents[int(batch_user[i])].memory_1.append(self.user_agents[int(batch_user[i])].update_memory[-1])
            batch_pos_item_descriptions.append(self.item_agents[int(batch_pos_item[i])].update_memory[-1])
            batch_neg_item_descriptions.append(self.item_agents[int(batch_neg_item[i])].update_memory[-1])

        batch_pos_item_descriptions_embeddings = [None for _ in range(len(batch_pos_item_descriptions))]
        batch_neg_item_descriptions_embeddings = [None for _ in range(len(batch_neg_item_descriptions))]
        if self.config['evaluation'] == 'rag':
            batch_pos_item_descriptions_embeddings = self._generate_embedding(batch_pos_item_descriptions)
            batch_neg_item_descriptions_embeddings = self._generate_embedding(batch_neg_item_descriptions)

        for i in range(batch_size):
            self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = \
                batch_pos_item_descriptions_embeddings[i]
            self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = \
                batch_neg_item_descriptions_embeddings[i]
            
    def full_sort_predict(self, interaction, idxs): # 需要指导
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return: score
        """
        print("user special full sort predict function")
        batch_size = idxs.shape[0]
        batch_pos_item = interaction[self.ITEM_ID]
        # 这里是测试时给数据的流程
        # TODO: load previously saved user and item agents' memories
        if self.config['loaded']:
            self.record_idx = self.config['saved_idx']
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}', )
            with open(f'{path}/user', 'r') as f:
                f.readline()
                for line in f:
                    user, user_description = line.strip().split('\t')
                    user_id = self.user_token_id[user]
                    self.user_agents[user_id].memory_1.append(user_description)
            batch_user = interaction[self.USER_ID]
            # for i, user in enumerate(batch_user):
            #     print(int(user))
            #     print(self.user_id_token[int(user)])
            #     self.user_agents[int(user)].historical_interactions = np.load(
            #         f'{path}/user_embeddings_{self.user_id_token[int(user)]}.npy', allow_pickle=True).item()
            #     self.rec_agent.user_examples[int(user)] = np.load(
            #         f'{path}/user_examples_{self.user_id_token[int(user)]}.npy', allow_pickle=True).item()
            # for i, item in enumerate(range(self.n_items)):
            #     if os.path.exists(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy'):
            #         self.item_agents[int(item)].memory_embedding = np.load(
            #             f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy', allow_pickle=True).item()

        if self.config['saved'] and not self.config['loaded']:
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}', )
            if not os.path.exists(path):
                os.makedirs(path)
            for item_id, item_context in self.item_agents.items():
                np.save(f'{path}/item_embeddings_{self.item_id_token[item_id]}.npy',
                        item_context.memory_embedding)
            with open(f'{path}/user', 'w') as f:
                f.write('user_id:token\tuser_description:token_seq\n')
                for user_id, user_context in self.user_agents.items():
                    user_description = user_context.memory_1[-1]
                    f.write(str(self.user_id_token[user_id]) + '\t' + user_description.replace('\n', ' ') + '\n')
                    np.save(f'{path}/user_embeddings_{self.user_id_token[user_id]}.npy',
                            user_context.historical_interactions)
                    np.save(f'{path}/user_examples_{self.user_id_token[user_id]}.npy',
                            self.rec_agent.user_examples[int(user_id)]) # user_embedding也都save了

        all_candidate_idxs = set(idxs.view(-1).tolist()) # 
        untrained_candidates = []
        for item in range(1, self.n_items): # 这里是将所有物品提取
            if list(self.item_agents[item].memory_embedding.keys())[-1].startswith('The CD is called'):
                if item in all_candidate_idxs:
                    untrained_candidates.append(item)
        print(
            f"In the reranking stage, there are {len(set(all_candidate_idxs))} candidates in total. \n There are {len(untrained_candidates)} have not been trained.")
        print("!!!")

        batch_user = interaction['user_id']
        batch_user_descriptions = []
        batch_group_descriptions = []
        for i in range(batch_size):
            batch_user_descriptions.append(self.user_agents[int(batch_user[i])].memory_1[-1])
            batch_group_descriptions.append(self.user_agents[int(batch_user[i])].update_group_memory[-1])
        if self.config['evaluation'] == 'rag' and self.config['item_representation'] == 'rag':
            batch_user_embedding_description = self._generate_embedding(batch_user_descriptions)
        else:
            batch_user_embedding_description = None

        scores = torch.full((batch_user.shape[0], self.n_items), -10000.)
        user_descriptions, list_of_item_descriptions, candidate_texts, user_his_texts, batch_user_embedding_explanations, batch_user_his, batch_select_examples, batch_group_descriptions = [], [], [], [], [], [], None, []
        for i in range(batch_size):
            user_id = int(batch_user[i])
            user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description = self._get_batch_inputs(
                interaction, idxs, i, batch_user_embedding_description)
            
            user_descriptions.append(self.user_agents[user_id].memory_1[-1])

            batch_group_descriptions.append(self.user_agents[user_id].update_group_memory[-1]) # 这个不可实现，感觉有一些问题

            user_his_texts.append(user_his_text)
            list_of_item_descriptions.append('\n\n'.join(candidate_text_order_description))
            candidate_texts.append(candidate_text)
            batch_user_his.append(list(self.rec_agent.user_examples[user_id].keys()))

        if self.config['evaluation'] == 'rag':
            batch_select_examples = []
            query_embeddings = self._generate_embedding(list_of_item_descriptions)
            for i in tqdm(range(batch_size)):
                user_his_descriptions = self.user_agents[int(batch_user[i])].memory_1[1:-1]
                user_his_description_embeddings = self._generate_embedding(user_his_descriptions)
                distances = distances_from_embeddings(query_embeddings[i], user_his_description_embeddings)
                index = indices_of_nearest_neighbors_from_distances(distances)[0]
                batch_select_examples.append(user_his_descriptions[index])
            np.save(os.path.join(path, 'batch_select_examples.npy'), np.array(batch_select_examples))

        if self.config['evaluation'] != 'sequential':
            user_his_texts = None

        messages = self._evaluation(batch_user, user_descriptions, user_his_texts,
                                    list_of_item_descriptions,
                                    batch_group_descriptions, batch_select_examples)

        # # TODO: logging
        # for i in range(batch_size):
        #     user_id = int(batch_user[i])
        #     pos_item = int(batch_pos_item[i])
        #     path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
        #     with open(osp.join(path, f'user.{user_id}'), 'a') as f:
        #         f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
        #         f.write(
        #             f'The evaluation prompts are as follows: {evaluation_prompts[i]}\n\n')
        #
        #         f.write(f'The system results are as follows: {messages[i]} \n\n')
        #         f.write(
        #             f"The pos item is: {self.item_agents[pos_item].role_description['item_title']}. Its related descriptions is as follows: {list(self.item_agents[pos_item].memory_embedding.keys())[-1]} \n\n")

        batch_pos_item = interaction[self.ITEM_ID]

        self._parsing_output_text(scores, messages, idxs, candidate_texts, batch_pos_item)
        # 按照scores来进行排布
        return scores

    def _evaluation(self, batch_user, user_descriptions, user_his_texts, list_of_item_descriptions,
                    batch_group_descriptions, batch_select_examples=None,):
        batch_size = len(user_descriptions)
        if batch_select_examples is not None:
            # retrieval mode:
            # evaluation_prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [],
            #                                                       list_of_item_descriptions[i],
            #                                                       batch_select_examples[i]) for i in range(batch_size)]
            task_list = [self.rec_agent.evaluate(
                int(batch_user[i]), user_descriptions[i], [], list_of_item_descriptions[i], batch_select_examples[i]
            ) for i in range(batch_size)]
        else:
            if self.config['evaluation'] == 'sequential':
                # evaluation_prompts = [
                #     self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], user_his_texts[i],
                #                                     list_of_item_descriptions[i]) for i in range(batch_size)]
                task_list = [self.rec_agent.evaluate(
                    int(batch_user[i]), user_descriptions[i], user_his_texts[i], list_of_item_descriptions[i]
                ) for i in range(batch_size)]
            else:
                # evaluation_prompts = [
                #     self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [],
                #                                     list_of_item_descriptions[i]) for i in range(batch_size)]
                task_list = [self.rec_agent.evaluate_graph(
                    int(batch_user[i]), user_descriptions[i], [], list_of_item_descriptions[i], group_description=batch_group_descriptions[i],
                ) for i in range(batch_size)]

        loop = asyncio.get_event_loop()
        messages = loop.run_until_complete(asyncio.gather(*task_list))

        return messages
