# GraphKernelSVM

**GraphKernelSVM** は、グラフ構造データに対してサポートベクターマシン（SVM）を適用するための、Pythonベースのグラフカーネル実装です。  
主に **Weisfeiler-Lehman（WL）サブツリーカーネル** を用いたグラフ分類が可能です。

---

## 特徴

- Weisfeiler-Lehman（WL）カーネルの実装
- `scikit-learn`、`networkx` を利用した柔軟な構成
- 化学分子解析・ソーシャルグラフ・生体ネットワークなどへの応用が可能

---

## 使用データセット（データ出典）
本プロジェクトでは、Twitterボット検出のためのグラフデータセット TwiBot-22 の c-15 を使用しています。
このデータセットは以下の論文に基づいて提供されています：

@inproceedings{fengtwibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and others},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}

TwiBot-22は、ユーザー間のフォロー関係や投稿内容をグラフ構造として表現した大規模データセットであり、グラフカーネルを用いたボット検出のベンチマークとして活用されています。

## インストール方法

まずはリポジトリをクローンしてください：

```bash
git clone https://github.com/kkkk-ui/GraphKernelSVM.git
cd GraphKernelSVM
