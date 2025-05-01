import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#============================================================================================#
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
        elif source_label not in {'human', 'bot'}:
            if source_id not in node:
                node[source_id] = source_label
                G.add_node(source_id, label=source_label)
            continue
        elif target_label not in {'human', 'bot'}:
            if target_id not in node:
                node[target_id] = target_label
                G.add_node(target_id, label=target_label)
            continue
        
        # エッジを追加
        if source_id in edge:
            edge[source_id].append(target_id)
            G.add_edge(source_id, target_id)
        else:
            edge[source_id]=[]
            edge[source_id].append(target_id)
            G.add_edge(source_id, target_id)

    count_human = sum(1 for label in node.values() if label == "human")
    print(f"labelがhumanの数: {count_human}")
    count_bot = sum(1 for label in node.values() if label == "bot")
    print(f"labelがbotの数: {count_bot}")
    print(f"labelがuserの数: {count_bot+count_human}")
    count_unknown = sum(1 for label in node.values() if label == "unknown")
    print(f"labelがunknownの数: {count_unknown}")
    
    # 格納用リスト
    human_nodes = []
    bot_nodes = []

    # ラベル情報を辞書に格納
    for row in split_f:
        split[row['id']] = row['split']
    
    # 条件に基づいて各リストに最大100個ずつ追加
    for key, value in split.items():
        if node[key] == 'human' and len(human_nodes) < 50:
            human_nodes.append(key)
        elif node[key] == 'bot' and len(bot_nodes) < 50:
            bot_nodes.append(key)
            
    # numpy配列に変換
    human_nodes = np.array(human_nodes)
    bot_nodes = np.array(bot_nodes)

    vertification_nodes = np.concatenate((human_nodes,bot_nodes))
#============================================================================================#

#============================================================================================#
# 初期化してファイルに書き込む
with open('subgraph_detail_log.txt', 'w') as file:
    file.write("label,human_nodes,bot_nodes,max_degree\n")
for center_node in vertification_nodes:
    # サブグラフ抽出
    k = 2

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
    
    human_labels = 0
    bot_labels = 0

    sublabel = {node_id: label for i, (node_id, label) in enumerate(subgraph.nodes(data="label"))}
    edge_list = [(u, v) for u, v in subgraph.edges()]
    
    # 各ノードの次数を取得
    degrees = dict(subgraph.degree())
    degrees_without_center = {node: deg for node, deg in degrees.items() if node != center_node}


    # 最大次数を持つノードとその次数を表示
    max_degree_node = max(degrees_without_center, key=degrees_without_center.get)
    max_degree = degrees_without_center[max_degree_node]
    

    node_colors = ["red" if node == center_node 
                   else ("k" if sublabel[node] == "human" else "blue")
                   for node in subgraph.nodes]

    for node in subgraph.nodes:
        if sublabel[node] == 'human':
            human_labels += 1
        else:
            bot_labels += 1
    with open('subgraph_detail_log.txt', 'a') as file:
        file.write(f"{sublabel[center_node]},{human_labels},{bot_labels},{max_degree}\n")
    
    # サブグラフプロット
    plt.figure(figsize=(10, 8), dpi=300) 
    nx.draw_networkx(subgraph,
                    node_size = 0.5,
                    node_color = node_colors,
                    edgecolors = 'gray', # node border color
                    linewidths = 0.05, # node border width
                    with_labels = False,
                    edge_cmap = plt.cm.RdBu_r,
                    width = 0.01)

    # 黒い枠（スプライン）を消す
    ax = plt.gca()  # 現在の軸を取得
    for spine in ax.spines.values():  # スプライン（外枠）を非表示
        spine.set_visible(False)


    plt.text(
        1.0, 1.05,  # 右上（軸座標系）
        f"Center Node Info:\nnode_id: {center_node}\nlabel: {sublabel[center_node]}",
        fontsize=10,  # テキストサイズ
        ha="right",  # 横方向で右揃え
        va="bottom",  # 縦方向で下揃え
        transform=plt.gca().transAxes,  # 軸座標系に基づく位置指定
    )
    plt.show()

#============================================================================================#
