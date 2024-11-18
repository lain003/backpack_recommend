from django.core.management.base import BaseCommand

import os
import pprint
import tempfile

from buildcrawler.models import Item, Video, Build, BattleSet, Round, Recipe
from django.db import connection

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

class Command(BaseCommand):
    help = ""

    class MovielensModel(tfrs.Model):
        def __init__(self, user_model, movie_model, task):
            super().__init__()
            self.movie_model: tf.keras.Model = movie_model
            self.user_model: tf.keras.Model = user_model
            self.task: tfrs.tasks.Retrieval = task

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features["user_id"])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.movie_model(features["movie_title"])

            # The task computes the loss and the metrics.
            return self.task(user_embeddings, positive_movie_embeddings)
    
    def hoge1(self):
        # Ratings data.
        ratings = tfds.load("movielens/100k-ratings", split="train")
        # Features of all the available movies.
        movies = tfds.load("movielens/100k-movies", split="train")
        for x in ratings.take(1).as_numpy_iterator():
            pprint.pprint(x)

        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        })
        movies = movies.map(lambda x: x["movie_title"])


        # 学習用とテスト用のデータ分割
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)


        movie_titles = movies.batch(1_000)
        user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
        unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
        unique_user_ids = np.unique(np.concatenate(list(user_ids)))

        # クエリータワー
        embedding_dimension = 32
        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            # We add an additional embedding to account for unknown tokens.
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # 候補タワー
        movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # メトリクス
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(movie_model)
        )

        # 損失
        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        model = Command.MovielensModel(user_model, movie_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        cached_train = train.shuffle(100_000).batch(8192).cache()
        cached_test = test.batch(4096).cache()
        #モデル訓練
        model.fit(cached_train, epochs=3)
        #モデル評価
        model.evaluate(cached_test, return_dict=True)

        # 予測を立てる
        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
            tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
        )
        # Get recommendations.
        _, titles = index(tf.constant(["42"]))
        # import pdb; pdb.set_trace()
        print(f"Recommendations for user 42: {titles[0, :3]}")

    def hoge2(self):
        from collections import defaultdict
        def dictfetchall(cursor):
            """
            Return all rows from a cursor as a dict.
            Assume the column names are unique.
            """
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        with connection.cursor() as cursor:
            cursor.execute('''SELECT id, num, battle_set_id
                                FROM (
                                    SELECT id, num, battle_set_id,
                                        ROW_NUMBER() OVER (PARTITION BY battle_set_id ORDER BY num DESC, id DESC) AS row_num
                                    FROM buildcrawler_round
                                ) sub
                                WHERE row_num = 1;'''
                           )
            rounds = dictfetchall(cursor)
        round_ids = [x["id"] for x in rounds]
        builds = Build.objects.filter(round__in=round_ids)
        grouped_builds = defaultdict(list)
        for build in builds:
            grouped_builds[build.round_id].append(build)
        
        final_round_mixed_items = {}
        for round_id, builds in grouped_builds.items():
            mixed_items = []
            for build in builds:
                items = []
                for _ in range(build.num):
                    items.append(build.item)
                #mixed_items += items
                mixed_items += Item.get_mixed_items(items)
            final_round_mixed_items[round_id] = mixed_items

        import pandas as pd
        
        item_master = Item.objects.order_by("id")
        item_index = {key.id: index + 1 for index, key in enumerate(item_master)}
        max_item_num = 10
        
        item_logs_matrix = []
        
        for round_id, items in final_round_mixed_items.items():
            sort_items = sorted(items, key=lambda x: x.id)

            matrix_row = [0 for _ in range(0, len(item_master) * max_item_num)]
            before_item_id = 0
            num = 0
            for item in sort_items:
                if before_item_id != item.id:
                    num = 0
                else:
                    num += 1
                    if num >= max_item_num:
                        print("max_item_num over")
                        num -= 1
                matrix_row[(item_index[item.id] - 1) * max_item_num + num] = 1
                before_item_id = item.id

            item_logs_matrix.append(matrix_row)

        indexs = []
        for i in item_master:
            for j in range(1, max_item_num+1):
                indexs.append(f'{i.name}_{j}')
        df_item_logs = pd.DataFrame(np.array(item_logs_matrix).T, index=indexs)
        import pandas as pd
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.frequent_patterns import association_rules
        pd.set_option('display.max_colwidth', None)
        # min_item_frequency = 5
        # filtered_df = df_item_logs.loc[:, (df_item_logs.sum(axis=0) >= min_item_frequency)]
        # Aprioriアルゴリズムを使って頻出アイテムセットを抽出 max_lenを設定することで時間を短縮できる
        frequent_itemsets = fpgrowth(df_item_logs.T.astype('bool'), min_support=0.2, use_colnames=True, max_len=5)
        # アソシエーションルールの抽出（confidence >= 0.1に設定）
        rules = association_rules(frequent_itemsets, min_threshold=0.1, metric="confidence")
        itemset_3more = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 3)].sort_values(by='support', ascending=False)
        #rules[rules['antecedents'].apply(lambda x,target_items=target_items: set(target_items) == set(x)) & rules['consequents'].apply(lambda x: len(x) == 1)].sort_values(by='lift', ascending=False)[0:20]
        target_items = ['WoodenSword_1','LuckyClover_1']
        Item.get_evolve_item(Item.objects.get(name="WoodenSword"))
        filtered_rules = rules[rules['antecedents'].apply(lambda x,target_items=target_items: all(item in x for item in target_items))]
        rules[rules['antecedents'].apply(lambda x,target_items=target_items: set(target_items) == set(x))]
        
        # confidenceでソート（降順）
        sorted_rules = rules.sort_values(by='support', ascending=False)
        import pdb; pdb.set_trace()
        # 結果を表示
        print("頻出アイテムセット:")
        print(frequent_itemsets.sort_values(by='support', ascending=False))
        print("\nアソシエーションルール:")
        print(sorted_rules)

        def gmm():
            from sklearn.mixture import GaussianMixture
            import matplotlib.pyplot as plt
            gmm = GaussianMixture(n_components=20)
            gmm.fit(df_item_logs)

            item_clusters = gmm.predict_proba(df_item_logs)
            item_prob_df = pd.DataFrame(item_clusters, index=indexs, columns=[f'Cluster_{i}' for i in range(gmm.n_components)])
            item_prob_df.loc["Banana_1"]
            item_prob_df.loc[:, "Cluster_1"]

            cluster_name = "Cluster_3"
            item_prob_df.loc[item_prob_df[cluster_name] != 0, cluster_name]

            # 各アイテムが最も高い確率で所属するクラスターを取得
            item_clusters = np.argmax(item_clusters, axis=1)

            # クラスタリング結果を図示
            plt.figure(figsize=(16, 12))

            # 各アイテムをクラスターに応じて色分けしてプロット
            for i, (item, cluster) in enumerate(zip(indexs, item_clusters)):
                plt.scatter(i, cluster, s=100, label=item, cmap='viridis')

            # グラフの詳細設定
            plt.title('GMM Clustering of Items')
            plt.xlabel('Items')
            plt.ylabel('Clusters')
            plt.xticks(np.arange(len(indexs)), indexs, rotation=45)
            plt.yticks(np.arange(20),[f'Cluster_{i}' for i in range(gmm.n_components)])

            # グリッドと凡例を表示
            plt.grid(True)
            plt.legend(loc='best', title='Items')
            plt.show()
    
    def hoge3(self):
        import pandas as pd
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules
        import networkx as nx
        import matplotlib.pyplot as plt

        # サンプルデータ: ユーザー×アイテムの購入履歴（1: 購入, 0: 未購入）
        data = {
            'user_1': [1, 0, 1, 0, 1],
            'user_2': [0, 1, 1, 1, 0],
            'user_3': [1, 1, 0, 0, 1],
            'user_4': [0, 0, 1, 1, 0],
            'user_5': [1, 0, 0, 1, 1]
        }

        # DataFrameに変換
        df = pd.DataFrame(data, index=['item_A', 'item_B', 'item_C', 'item_D', 'item_E']).T

        # Aprioriアルゴリズムで頻出アイテムセットを抽出（min_support = 0.5）
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

        # アソシエーションルールを生成
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

        # NetworkXグラフを作成
        G = nx.DiGraph()

        # ルールからノードとエッジを追加
        for _, rule in rules.iterrows():
            antecedents = tuple(rule['antecedents'])  # 前提（antecedent）
            consequents = tuple(rule['consequents'])  # 結果（consequent）
            
            # ノードを追加
            G.add_node(antecedents)
            G.add_node(consequents)
            
            # エッジを追加（ルールに従って、前提→結果の方向）
            G.add_edge(antecedents, consequents, weight=rule['confidence'])

        # グラフの描画
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, k=1.5, seed=42)  # レイアウト設定

        # ノードとエッジの描画
        nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, font_color='black', node_color='lightblue', arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # グラフを表示
        plt.title('Association Rule Graph')
        plt.show()
  
    def handle(self, *args, **options):
        # hoge1()
        self.hoge2()