from django.core.management.base import BaseCommand

from buildcrawler.models import Item, Video, Build, BattleSet, Round, Recipe
from buildcrawler.logic.backpack_cv import BackPackCV

import cv2
import os
import numpy as np

import os.path
import mss

class Command(BaseCommand):
    help = ""

    def outputimg_results(self, img, item_match_results):
        for item_match_result in item_match_results:
            print(f"{item_match_result.item} {len(item_match_result.box_and_scores)}")
            # Build.objects.create(item=item_match_result.item, num=len(item_match_result.box_and_scores), player=2)
            for box_and_score in item_match_result.box_and_scores:
                box = box_and_score.box
                cv2.putText(img,item_match_result.item.name, ( box[0], box[1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
                cv2.rectangle(img, (box[0], box[1] + 5), (box[2], box[3]), (255, 0, 0), 3)
        cv2.imshow("After NMS", cv2.resize(img,None,fx=1,fy=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def analyze_shop_item(self, screenshot_name) -> tuple[list[Item], list[Item]]:
        # 写っている画面でとるなら
        # with mss.mss() as sct:
        #     sct.shot(output="recommend_screenshot.png")

        n = np.fromfile(f"{screenshot_name}.png", np.uint8)
        origin_screen_img = cv2.imdecode(n, cv2.IMREAD_COLOR)

        screen_img = cv2.resize(origin_screen_img, dsize=(960, 540))
        build_img = screen_img[0 : 320, 0 : 415]
        shop_img = screen_img[0 : 397, 650 : 950]

        build_items = []
        for item_natch_result in BackPackCV.identify_items(build_img, False):
            for _ in item_natch_result.box_and_scores:
                build_items.append(item_natch_result.item)

        shop_items = []
        for item_natch_result in BackPackCV.identify_items(shop_img, False):
            for _ in item_natch_result.box_and_scores:
                shop_items.append(item_natch_result.item)

        print("--------Have Item--------")
        for item in build_items:
            print(item.name)
        print("--------Shop Item--------")
        for item in shop_items:
            print(item.name)

        return build_items, shop_items
    
    # # 複雑すぎるのでいったんコメントアウト
    # # return [score, match_index]
    # def search_mixed_items(self, search_item: Item, builds: list[Build], ignore_builds_index: list[int]) -> Tuple[int, list[int]]:
    #     recipes = Recipe.objects.filter(result_item=search_item).select_related("mixed_item")
    #     if len(recipes) == 0:
    #         return []
        
    #     mixed_items = []
    #     for recipe in recipes:
    #         for _ in range(recipe.num):
    #             mixed_items.append(recipe.mixed_item)

    #     score = 0
    #     match_index = []
    #     for mixed_item in mixed_items:
    #         for index, build in enumerate(builds):
    #             if index in ignore_builds_index: next 
    #             if build.item_id == mixed_item.id:
    #                 score += mixed_item.gold
    #                 match_index = index

    #top_scores [(4_1, 24),....]
    def show_topscore(self, top_scores):
        presented_battle_sets = {}
        no_duplicate_top_scores = []
        for top_score in top_scores:
            round_id = top_score[0].split("_")[0]
            # N+1
            battle_set_id = Round.objects.get(id=round_id).battle_set_id
            if battle_set_id in presented_battle_sets.keys():
                next
            else:
                presented_battle_sets[battle_set_id] = ""
                no_duplicate_top_scores.append(top_score)

        print(no_duplicate_top_scores)

        def concat_tile(im_list_2d):
            return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
        
        images = []
        for top_score in no_duplicate_top_scores:
            round_id = top_score[0].split("_")[0]
            round = Round.objects.get(id=round_id)
            set_rounds = Round.objects.filter(battle_set_id=round.battle_set_id)

            img = cv2.imread(f'buildcrawler/images/result/{top_score[0]}.png')
            cv2.putText(img, f"id:{round.id}, num:{round.num} len:{len(set_rounds)}", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3, cv2.LINE_AA)
            images.append(img)

        im_tile = concat_tile([[images[0], images[1], images[2], images[3]],
                                [images[4], images[5], images[6], images[7]],
                                [images[8], images[9], images[10], images[11]]])
        
        cv2.imshow("After NMS", cv2.resize(im_tile,None,fx=0.9,fy=0.9))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_topscore(self, have_items: list[Item], sell_items: list[Item], from_round: int, to_round: int, player: int, multiplier_sell_score: float) -> dict[int]:
        def build_key(build):
            return f"{build.round.id}_{build.player}"

        have_items_mixed = Item.get_mixed_items(have_items)
        for item in have_items_mixed:
            print(item.name)
        print("--------------")
        sell_items_mixed = Item.get_mixed_items(sell_items)
        for item in sell_items_mixed:
            print(item.name)

        # accept機能をいれるならここにfilter
        rounds = Round.objects.filter(num__gte=from_round, num__lte=to_round)
        builds = Build.objects.filter(round__in=rounds, player__in=player)

        builds_items = {}
        for build in builds:
            items = []
            for _ in range(build.num):
                items.append(build.item)
            
            mixed_items = Item.get_mixed_items(items)
            key = build_key(build)
            if key in builds_items:
                builds_items[key].extend(mixed_items)
            else:
                builds_items[key] = mixed_items

        # これだと合成済みの物の素材を検索してしまう不具合がある。
        scores = {}
        for build_item_key, build_items in builds_items.items():
            score = 0
            match_indexs = []
            for shop_item in have_items_mixed:
                for index, build_item in enumerate(build_items):
                    if index in match_indexs: next
                    if shop_item.id == build_item.id:
                        #print(f"{build_item_key}_{shop_item.name}_{shop_item.gold}_{score}")
                        score += shop_item.gold
                        match_indexs.append(index)
                        break
            for shop_item in sell_items_mixed:
                for index, build_item in enumerate(build_items):
                    if index in match_indexs: next
                    if shop_item.id == build_item.id:
                        #print(f"{build_item_key}_{shop_item.name}_{shop_item.gold}_{score}")
                        score += int(shop_item.gold * multiplier_sell_score)
                        match_indexs.append(index)
                        break
            scores[build_item_key] = score

        return scores
    
    def analyze(self, have_items, sell_items):
        from collections import defaultdict
        from django.db import connection
        # Lastround
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
        # round_ids = [x.id for x in Round.objects.filter(num__in=[5,6,7,8,9,10,11])]
        
        builds = Build.objects.filter(round__in=round_ids,player=1)
        grouped_builds = defaultdict(list)
        for build in builds:
            grouped_builds[build.round_id].append(build)
        final_round_mixed_items = {}
        for round_id, g_builds in grouped_builds.items():
            mixed_items = []
            for build in g_builds:
                items = []
                for _ in range(build.num):
                    items.append(build.item)
                #mixed_items += items
                mixed_items += Item.get_mixed_items(items)
            final_round_mixed_items[round_id] = mixed_items
        
        final_round_mixed_items[-1] = Item.get_mixed_items(have_items)
        #final_round_mixed_items[-1] = have_items
        import pandas as pd
        
        # item_masterとmax_item_numの設定
        item_master = Item.objects.order_by("id")
        max_item_num = 10

        # 空のDataFrameを初期化
        df_item_logs = pd.DataFrame()

        # 各ラウンドのアイテム処理
        for round_id, items in final_round_mixed_items.items():
            sort_items = sorted(items, key=lambda x: x.id)
            
            # デフォルトの行データを作成
            matrix_row = {f'{i.name}_{j}': 0 for i in item_master for j in range(1, max_item_num + 1)}
            before_item_id = None
            num = 0

            # ソートしたアイテムの処理
            for item in sort_items:
                num = num + 1 if before_item_id == item.id else 1
                if num > max_item_num:
                    print("max_item_num over")
                    num = max_item_num

                # 行データにアイテム情報をセット
                matrix_row[f'{item.name}_{num}'] = 1
                before_item_id = item.id

            # 行をDataFrameに追加
            line = pd.DataFrame(matrix_row, index=[round_id])
            df_item_logs = pd.concat([df_item_logs, line], ignore_index=False)
        
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity

        # ユーザー間のコサイン類似度を計算
        user_similarity = pd.DataFrame(cosine_similarity(df_item_logs), index=df_item_logs.index, columns=df_item_logs.index)
        user = -1
        df = df_item_logs.T
        similar_users = user_similarity[user].sort_values(ascending=False)
        #filter_users = similar_users[similar_users > 0.5]
        filter_users = similar_users[0:10]
        filter_df = df[filter_users.index.to_list()]
        
        # Userが持っていないアイテムを確認し、他のユーザーが持っている数をカウント
        items_not_owned_by_user1 = filter_df.index[filter_df[user] == 0].tolist()
        items_count = {}
        for item in items_not_owned_by_user1:
            sum = filter_df.loc[item].sum()
            if sum != 0:
                items_count[item] = filter_df.loc[item].sum()

        items_count = dict(sorted(items_count.items(), key=lambda item: item[1], reverse=True))
        # import pdb; pdb.set_trace()
        # User1に対する推薦アイテムを取得
        # recommendations = recommend_items(df_item_logs.index[1], df_item_logs.T, user_similarity)

        def association(df_item_logs):
            import pandas as pd
            from mlxtend.frequent_patterns import apriori
            from mlxtend.frequent_patterns import fpgrowth
            from mlxtend.frequent_patterns import association_rules
            pd.set_option('display.max_colwidth', None)
            # min_item_frequency = 5
            # filtered_df = df_item_logs.loc[:, (df_item_logs.sum(axis=0) >= min_item_frequency)]
            # Aprioriアルゴリズムを使って頻出アイテムセットを抽出 max_lenを設定することで時間を短縮できる
            frequent_itemsets = fpgrowth(df_item_logs.astype('bool'), min_support=0.05, use_colnames=True, max_len=3)
            frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: {"Manathirst_1"}.issubset(x))]
            # アソシエーションルールの抽出（confidence >= 0.1に設定）
            rules = association_rules(frequent_itemsets, min_threshold=0.05, metric="confidence")
            itemset_3more = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 3)].sort_values(by='support', ascending=False)
            #rules[rules['antecedents'].apply(lambda x,target_items=target_items: set(target_items) == set(x)) & rules['consequents'].apply(lambda x: len(x) == 1)].sort_values(by='lift', ascending=False)[0:20]
            target_items = {'WoodenSword_1','LumpofCoal_1'}
            Item.get_evolve_item(Item.objects.get(name="WoodenSword"))
            filtered_rules = rules[rules['antecedents'].apply(lambda x,target_items=target_items: target_items == x)].sort_values(by='lift', ascending=False)
            rules[rules['antecedents'].apply(lambda x,target_items=target_items: target_items == x) & rules['consequents'].apply(lambda x: len(x) == 1)].sort_values(by='lift', ascending=False)[0:10] 
            rules[rules['antecedents'].apply(lambda x,target_items=target_items: set(target_items) == set(x))]

            shop_items = have_items + sell_items

            lift_means = {}
            # 各アイテムについて、antecedentsかconsequentsに含まれているかを確認し、対応するliftを取得
            for item in df_item_logs.columns:
                lifts = rules[rules['antecedents'].apply(lambda x: item in x) | rules['consequents'].apply(lambda x: item in x)]['lift']
                lift_means[item] = lifts.mean() if not lifts.empty else 0  # 平均liftを計算、空の場合は0
            dict(sorted(lift_means.items(), key=lambda item: item[1], reverse=True))
            import pdb; pdb.set_trace()

            def get_serial_item_names(items):
                # 出現回数をカウントする辞書
                counts = {}
                # 配列の要素に番号を付ける
                mix_items = []
                for item in Item.get_mixed_items(items):
                    if item not in counts:
                        counts[item] = 1
                    else:
                        counts[item] += 1
                    mix_items.append(f"{item.name}_{counts[item]}")
                return mix_items

            def get_array_combinations(shop_items):
                shop_mix_items = get_serial_item_names(shop_items)
                
                import itertools
                shop_mix_items_combinations = []
                for r1 in range(1, len(shop_mix_items)+1):
                    for combo1 in itertools.combinations(shop_mix_items, r1):
                        remaining = [item for item in shop_mix_items if item not in combo1]
                        # 残った配列の要素から選ぶ
                        for r2 in range(1, min(len(shop_mix_items)+1, len(remaining) + 1)):
                            for combo2 in itertools.combinations(remaining, r2):
                                shop_mix_items_combinations.append((list(combo1), list(combo2)))
                return shop_mix_items_combinations
            
            # shop_mix_items_combinations = get_array_combinations(shop_items)

            # filters = pd.DataFrame()
            # for shop_mix_items_combination in shop_mix_items_combinations:
            #     antecedents_target_items = shop_mix_items_combination[0]
            #     consequents_target_items = shop_mix_items_combination[1]
            #     filter_rules = rules[rules['antecedents'].apply(lambda x,target_items=antecedents_target_items: set(target_items) == set(x)) 
            #                          & rules['consequents'].apply(lambda x,target_items=consequents_target_items: set(target_items) == set(x))
            #                          & rules['lift'].apply(lambda x: x >= 1.1)
            #                          ].sort_values(by='lift', ascending=False)[0:20]
            #     if len(filter_rules) != 0:
            #         if filters.empty:
            #             filters = filter_rules
            #         else:
            #             filters = pd.concat([filters, filter_rules], ignore_index=True)
            # hoge = filters.sort_values(by='lift', ascending=False)[0:40]


            # have_items_serial = get_serial_item_names(have_items)
            # sell_items_serial = set(get_serial_item_names(shop_items)) - set(have_items_serial)
            # def get_cobinations(array):
            #     import itertools
            #     all_combinations = []
            #     for r in range(1, len(array) + 1):
            #         all_combinations.extend(itertools.combinations(array, r))
            #     return all_combinations
            # have_item_combinations = get_cobinations(have_items_serial)
            # popular_filters = rules[rules['antecedents'].apply(lambda x,target_items=have_item_combinations: set(x) in [set(tc) for tc in target_items])
            #                         & rules['consequents'].apply(lambda x: len(x) == 1)
            #                         & rules['consequents'].apply(lambda x,target_items=sell_items_serial: any(element in x for element in target_items))
            #                         ].sort_values(by='support', ascending=False)
            # aaa = rules[rules['antecedents'].apply(lambda x,target_items=have_item_combinations: set(x) in [set(tc) for tc in target_items]) 
            #     & rules['antecedents'].apply(lambda x: len(x) >= 2) 
            #     & rules['consequents'].apply(lambda x: len(x) == 1)].sort_values(by='lift', ascending=False)

            # bbb = pd.Series([item for sublist in aaa['consequents'][0:int(len(aaa)/2)] for item in sublist]).value_counts()
            # ccc = pd.Series([item for sublist in aaa['consequents'][int(len(aaa)/2):-1] for item in sublist]).value_counts()
            # # 結果を表示
            # print("頻出アイテムセット:")
            # print(frequent_itemsets.sort_values(by='support', ascending=False)[0:40])
            # print("\nアソシエーションルール:")
        association(df_item_logs)
        import pdb; pdb.set_trace()
    


    def screenshot_window(self, window_name, png_name):
        import win32gui
        import win32ui
        import win32con
        import win32api
        from PIL import Image
        # ウィンドウのハンドルを取得
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd == 0:
            print("指定されたウィンドウが見つかりません")
            return

        # ウィンドウの座標を取得
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        width = width*2
        height = height*2

        # ウィンドウデバイスコンテキストを取得
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # 画像オブジェクトを作成
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # バックグラウンドのウィンドウのスクリーンショットを取得
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        # スクリーンショットを保存
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )
        img.save(f"{png_name}.png")

        # 後処理
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
    
    
    def show_match_build(self, from_round, to_round, player, multiplier_sell_score):
        self.screenshot_window("Backpack Battles", "recommend_screenshot")
        have_items, sell_items = self.analyze_shop_item("recommend_screenshot")
        self.analyze(have_items, sell_items)
        #scores = self.get_topscore(have_items, sell_items, from_round, to_round, player, multiplier_sell_score)
        #top_scores = sorted(scores.items(), key=lambda x:-x[1])[0: 100]
        #self.show_topscore(top_scores)

    def handle(self, *args, **options):
        self.show_match_build(options["from_round"], options["to_round"], options["player"], options["multiplier_sell_score"])

    def add_arguments(self, parser):
        parser.add_argument('--from_round', nargs='?', default=1, type=int)
        parser.add_argument('--to_round', nargs='?', default=18, type=int)
        parser.add_argument('--player', nargs='?', default=[1,2], type=list)
        parser.add_argument('--multiplier_sell_score', nargs='?', default=0.5, type=float)