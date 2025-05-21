import networkx as nx
import numpy as np
import csv
from grakel import Graph

class classification:
    #-----------------------------------------------------------------------------------------#
    # パス、ラベルの設定
    def __init__(self,negLabel=-1,posLabel=1):
        self.negLabel = negLabel
        self.posLabel = posLabel
    #-----------------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------------#
    #グラフオブジェクトの作成
    def makeData(self,dataType=1):
        self.dataType = dataType
        self.graphs = []
        if dataType == 1:    
            # オリジナルグラフ作成
            with open("edge.csv", "r") as edgefile, open("label.csv", "r") as labelfile, open("split.csv", "r") as splitfile:
                edge_f = csv.DictReader(edgefile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                label_f = csv.DictReader(labelfile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                split_f = csv.DictReader(splitfile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

                label_dict = {}
                node = {}
                edge = {}
                split = {}
                G = nx.Graph()  

                # ラベル情報を辞書に格納
                for row in label_f:
                    label_dict[row['id']] = row

                # エッジ情報を処理
                for row in edge_f:
                    source_id = row['source_id']
                    target_id = row['target_id']
                    relation = row['relation']
                    source_label = ''
                    target_label = ''

                    # relationを確認してfollowもしくはfriendの場合のみ処理
                    if relation == 'follow' or relation == 'friend':
                        pass
                    else:
                        continue

                    # ラベルを持たない場合humanとする
                    try:
                        source_label = label_dict[source_id]['label']
                    except KeyError:
                        source_label = 'human'

                    try:
                        target_label = label_dict[target_id]['label']
                    except KeyError:
                        target_label = 'human'

                    # unknownを無視する場合
                    # source_id,target_idのラベルを確認してhumanまたはbotの場合のみ処理
                    if source_label not in {'human', 'bot'} and target_label not in {'human', 'bot'}:
                        continue
                    elif source_label in {'human', 'bot'} and target_label in {'human', 'bot'}:
                        # ノードを追加　既にノードとして存在しない場合に追加
                        if source_id not in node:
                            node[source_id] = source_label
                            G.add_node(source_id, label=source_label)
                        if target_id not in node:
                            node[target_id] = target_label
                            G.add_node(target_id, label=target_label)
                    elif source_label in {'human', 'bot'}:
                        if source_id not in node:
                            node[source_id] = source_label
                            G.add_node(source_id, label=source_label)
                            continue
                        continue
                    elif target_label in {'human', 'bot'}:
                        if target_id not in node:
                            node[target_id] = target_label
                            G.add_node(target_id, label=target_label)
                            continue
                        continue
        
                    # エッジを追加
                    if source_id in edge:
                        edge[source_id].append(target_id)
                        G.add_edge(source_id, target_id)
                    else:
                        edge[source_id]=[]
                        edge[source_id].append(target_id)
                        G.add_edge(source_id, target_id)

                # ラベルのカウント
                count_human = sum(1 for label in node.values() if label == "human")
                print(f"labelがhumanの数: {count_human}")
                count_bot = sum(1 for label in node.values() if label == "bot")
                print(f"labelがbotの数: {count_bot}")
                print(f"labelがuserの数: {count_bot+count_human}")
                count_unknown = sum(1 for label in node.values() if label == "unknown")
                print(f"labelがunknownの数: {count_unknown}")
                # ここまでオリジナルグラフの作成
        
                # ラベル情報を辞書に格納
                for row in split_f:
                    split[row['id']] = row['split']

                # 学習データと評価データを作成
                Xtr = np.array([key for key, value in split.items() if value == 'train'])
                Ytr = np.array([label_dict[key]['label'] for key, value in split.items() if value == 'train'])
                label = []
                # ラベル付け
                for y in Ytr:
                    if y =='human':
                        label.append(1)
                    elif y =='bot':
                        label.append(-1)
                Ytr = np.array([i for i in label])

                Xte = np.array([key for key, value in split.items() if value == 'test' or value == 'val'])
                Yte = np.array([label_dict[key]['label'] for key, value in split.items() if value == 'test' or value == 'val'])
                label = []
                for y in Yte:
                    if y =='human':
                        label.append(1)
                    elif y =='bot':
                        label.append(-1)
                Yte = np.array([i for i in label])

                # 孤立ノードではないインデックスを保持するためのリスト
                valid_Xtr = []
                valid_Ytr = []
                valid_Xte = []
                valid_Yte = []

                # グラフオブジェクトを格納
                for idx, (center_node, label) in enumerate(zip(np.concatenate([Xtr, Xte]), np.concatenate([Ytr, Yte]))):
                    # サブグラフ抽出
                    k = 1

                    # k-ホップ以内のサブグラフを抽出
                    try:
                        subgraph_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k).keys()
                        subgraph = G.subgraph(subgraph_nodes)
                    except MemoryError:
                        print(f"Node {center_node} forms an too big subgraph, skipping.")
                        continue

                    # サブグラフのノード数を取得(ノード数で制限)
                    num_nodes = subgraph.number_of_nodes()
                    if num_nodes > 5000:
                        print(f"Node {center_node} forms a too big subgraph, skipping.")
                        print(f"subgraph has {num_nodes} nodes")
                        continue

                    # 孤立ノードの場合はスキップ
                    if num_nodes == 1:
                        print(f"Node {center_node} forms an isolated subgraph, skipping.")
                        print(f"subgraph has {num_nodes} nodes")
                        continue

                    # ノードラベルを取得
                    sublabel = {node_id: label for i, (node_id, label) in enumerate(subgraph.nodes(data="label"))}
                    
                    # エッジリストを取得（networkxのサブグラフから直接取得）
                    edge_list = [(u, v) for u, v in subgraph.edges()]

                    # grakel用Graphオブジェクトを作成（隣接行列を介さない）
                    graph = Graph(edge_list, node_labels=sublabel)
                    self.graphs.append(graph)
                    print(f"Node {center_node} generated successfully")
                    print(f"subgraph has {num_nodes} nodes")

                    """
                    # サブグラフの隣接行列を取得
                    try:
                        adj_matrix_sub_dense = nx.to_numpy_array(subgraph,dtype=np.int8)
                        print(f"Node {center_node} generated successfully")
                    except np.core._exceptions._ArrayMemoryError:
                        print(f"Node {center_node} forms an too big subgraph, skipping.")
                        continue
                    except MemoryError:
                        print(f"Node {center_node} forms an too big subgraph, skipping.")
                        continue

                    # 孤立ノードの場合はスキップ
                    if adj_matrix_sub_dense.shape[0] == 1:
                        print(f"Node {center_node} forms an isolated subgraph, skipping.")
                        continue
                    """

                    # 孤立ノードでない場合、リストに追加
                    if idx < len(Xtr):  # 学習データの場合
                        valid_Xtr.append(center_node)
                        valid_Ytr.append(label)
                    else:  # テストデータの場合
                        valid_Xte.append(center_node)
                        valid_Yte.append(label)

                # 最後に孤立ノードを除去したリストで置き換え
                self.Xtr = np.array(valid_Xtr)
                self.Xtr = np.array([i+1 for i in range(self.Xtr.shape[0])])
                self.Ytr = np.array(valid_Ytr)
                
                start = self.Xtr[-1] + 1

                self.Xte = np.array(valid_Xte)
                self.Xte = np.array([i+start for i in range(self.Xte.shape[0])])
                self.Yte = np.array(valid_Yte)

                self.G_id = np.concatenate([np.array(valid_Xtr), np.array(valid_Xte)])
                self.classes = np.concatenate([np.array(valid_Ytr),np.array(valid_Yte)])

                # ラベルのカウント
                print("一定サイズのサブグラフのみ抽出後")
                count_human = sum(1 for label in np.concatenate([self.Yte,self.Ytr]) if label == 1)
                print(f"labelがhumanの数: {count_human}")
                count_bot = sum(1 for label in np.concatenate([self.Yte,self.Ytr]) if label == -1)
                print(f"labelがbotの数: {count_bot}")
                print(f"labelがuserの数: {count_bot+count_human}")  
                print(f"データの数:{len(self.graphs)}")
                
        if dataType == 2:    
            # オリジナルグラフ作成
            with open("edge.csv", "r") as edgefile, open("label.csv", "r") as labelfile, open("split.csv", "r") as splitfile:
                edge_f = csv.DictReader(edgefile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                label_f = csv.DictReader(labelfile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
                split_f = csv.DictReader(splitfile, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

                label_dict = {}
                node = {}
                edge = {}
                split = {}
                G = nx.Graph()  

                # ラベル情報を辞書に格納
                for row in label_f:
                    label_dict[row['id']] = row

                # エッジ情報を処理
                for row in edge_f:
                    source_id = row['source_id']
                    target_id = row['target_id']
                    relation = row['relation']
                    source_label = ''
                    target_label = ''

                    # relationを確認してfollowもしくはfriendの場合のみ処理
                    if relation == 'follow' or relation == 'friend':
                        pass
                    else:
                        continue

                    # ラベルを持たない場合unknownとする
                    try:
                        source_label = label_dict[source_id]['label']
                    except KeyError:
                        source_label = 'unknown'

                    try:
                        target_label = label_dict[target_id]['label']
                    except KeyError:
                        target_label = 'unknown'

                    if source_label not in {'human', 'bot', 'unknown'} and target_label not in {'human', 'bot', 'unknown'}:
                        continue
                    elif source_label in {'human', 'bot', 'unknown'} and target_label in {'human', 'bot', 'unknown'}:
                        # ノードを追加　既にノードとして存在しない場合に追加
                        if source_id not in node:
                            node[source_id] = source_label
                            G.add_node(source_id, label=source_label)
                        if target_id not in node:
                            node[target_id] = target_label
                            G.add_node(target_id, label=target_label)
        
        
                    # エッジを追加
                    if source_id in edge:
                        edge[source_id].append(target_id)
                        G.add_edge(source_id, target_id)
                    else:
                        edge[source_id]=[]
                        edge[source_id].append(target_id)
                        G.add_edge(source_id, target_id)

                # ラベルのカウント
                count_human = sum(1 for label in node.values() if label == "human")
                print(f"labelがhumanの数: {count_human}")
                count_bot = sum(1 for label in node.values() if label == "bot")
                print(f"labelがbotの数: {count_bot}")
                print(f"labelがuserの数: {count_bot+count_human}")
                count_unknown = sum(1 for label in node.values() if label == "unknown")
                print(f"labelがunknownの数: {count_unknown}")
                # ここまでオリジナルグラフの作成
        
                # ラベル情報を辞書に格納
                for row in split_f:
                    split[row['id']] = row['split']

                # 学習データと評価データを作成
                Xtr = np.array([key for key, value in split.items() if value == 'train'])
                Ytr = np.array([label_dict[key]['label'] for key, value in split.items() if value == 'train'])
                label = []
                # ラベル付け
                for y in Ytr:
                    if y =='human':
                        label.append(1)
                    elif y =='bot':
                        label.append(-1)
                Ytr = np.array([i for i in label])

                Xte = np.array([key for key, value in split.items() if value == 'test' or value == 'val'])
                Yte = np.array([label_dict[key]['label'] for key, value in split.items() if value == 'test' or value == 'val'])
                label = []
                for y in Yte:
                    if y =='human':
                        label.append(1)
                    elif y =='bot':
                        label.append(-1)
                Yte = np.array([i for i in label])

                # 孤立ノードではないインデックスを保持するためのリスト
                valid_Xtr = []
                valid_Ytr = []
                valid_Xte = []
                valid_Yte = []

                # グラフオブジェクトを格納
                for idx, (center_node, label) in enumerate(zip(np.concatenate([Xtr, Xte]), np.concatenate([Ytr, Yte]))):
                    # サブグラフ抽出
                    k = 1

                    # k-ホップ以内のサブグラフを抽出
                    try:
                        subgraph_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k).keys()
                        subgraph = G.subgraph(subgraph_nodes)
                    except MemoryError:
                        print(f"Node {center_node} forms an too big subgraph, skipping.")
                        continue

                    # サブグラフのノード数を取得(ノード数で制限)
                    num_nodes = subgraph.number_of_nodes()
                    if num_nodes > 10000:
                        print(f"Node {center_node} forms a too big subgraph, skipping.")
                        print(f"subgraph has {num_nodes} nodes")
                        continue

                    # 孤立ノードの場合はスキップ
                    if num_nodes == 1:
                        print(f"Node {center_node} forms an isolated subgraph, skipping.")
                        print(f"subgraph has {num_nodes} nodes")
                        continue

                    # ノードラベルを取得
                    sublabel = {node_id: label for i, (node_id, label) in enumerate(subgraph.nodes(data="label"))}
                    
                    # エッジリストを取得（networkxのサブグラフから直接取得）
                    edge_list = [(u, v) for u, v in subgraph.edges()]

                    # grakel用Graphオブジェクトを作成（隣接行列を介さない）
                    graph = Graph(edge_list, node_labels=sublabel)
                    self.graphs.append(graph)
                    print(f"Node {center_node} generated successfully")
                    print(f"subgraph has {num_nodes} nodes")


                    # 孤立ノードでない場合、リストに追加
                    if idx < len(Xtr):  # 学習データの場合
                        valid_Xtr.append(center_node)
                        valid_Ytr.append(label)
                    else:  # テストデータの場合
                        valid_Xte.append(center_node)
                        valid_Yte.append(label)

                # 最後に孤立ノードを除去したリストで置き換え
                self.Xtr = np.array(valid_Xtr)
                self.Xtr = np.array([i+1 for i in range(self.Xtr.shape[0])])
                self.Ytr = np.array(valid_Ytr)
                
                start = self.Xtr[-1] + 1

                self.Xte = np.array(valid_Xte)
                self.Xte = np.array([i+start for i in range(self.Xte.shape[0])])
                self.Yte = np.array(valid_Yte)

                self.G_id = np.concatenate([np.array(valid_Xtr), np.array(valid_Xte)])
                self.classes = np.concatenate([np.array(valid_Ytr),np.array(valid_Yte)])

                # ラベルのカウント
                print("一定サイズのサブグラフのみ抽出後")
                count_human = sum(1 for label in np.concatenate([self.Yte,self.Ytr]) if label == 1)
                print(f"labelがhumanの数: {count_human}")
                count_bot = sum(1 for label in np.concatenate([self.Yte,self.Ytr]) if label == -1)
                print(f"labelがbotの数: {count_bot}")
                print(f"labelがuserの数: {count_bot+count_human}")  
                print(f"データの数:{len(self.graphs)}")
    #-----------------------------------------------------------------------------------------#
