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

    def analyze_shop_item(self) -> tuple[list[Item], list[Item]]:
        with mss.mss() as sct:
            sct.shot(output="recommend_screenshot.png")

        # files = os.listdir("C:/Screenshots")
        # files = ["C:/Screenshots/" + i for i in files if i.endswith('.png') == True]
        # files.sort(reverse=True)
        # n = np.fromfile(files[0], np.uint8)
        n = np.fromfile("recommend_screenshot.png", np.uint8)
        origin_screen_img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        # 日本語のファイル名を取れないので使えない
        # origin_screen_img = cv2.imread(files[0])

        screen_img = cv2.resize(origin_screen_img, dsize=(960, 540))
        build_img = screen_img[0 : 320, 0 : 415]
        shop_img = screen_img[0 : 397, 650 : 950]

        build_items = BackPackCV.identify_items(build_img, False)
        shop_items = BackPackCV.identify_items(shop_img, False)
        return [x.item for x in build_items], [x.item for x in shop_items]
    
    def get_mixed_item(self,item) -> list[Item]:
        recipes = Recipe.objects.filter(result_item=item)
        if len(recipes) == 0:
            return []
        
        return_items = []
        for recipe in recipes:
            for _ in range(recipe.num):
                items = self.get_mixed_item(recipe.mixed_item)
                if len(items) != 0:
                    return_items.extend(items)
                else:
                    return_items.append(recipe.mixed_item)
        return return_items
    
    def get_mixed_items(self,items: list[Item]) -> list[Item]:
        return_items = []
        for item in items:
            mixed_items = self.get_mixed_item(item)
            if len(mixed_items) == 0:
                return_items.append(item)
            else:
                return_items.extend(mixed_items)
        return return_items
    
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
            images.append(cv2.imread(f'buildcrawler/images/result/{top_score[0]}.png'))

        im_tile = concat_tile([[images[0], images[1], images[2], images[3]],
                                [images[4], images[5], images[6], images[7]],
                                [images[8], images[9], images[10], images[11]]])
        
        cv2.imshow("After NMS", cv2.resize(im_tile,None,fx=0.9,fy=0.9))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_topscore(self, have_items: list[Item], sell_items: list[Item], from_round: int, to_round: int, player: int, multiplier_sell_score: float) -> dict[int]:
        def build_key(build):
            return f"{build.round.id}_{build.player}"

        have_items_mixed = self.get_mixed_items(have_items)
        for item in have_items_mixed:
            print(item.name)
        print("--------------")
        sell_items_mixed = self.get_mixed_items(sell_items)
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
            
            mixed_items = self.get_mixed_items(items)
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

    def handle(self, *args, **options):
        have_items, sell_items = self.analyze_shop_item()
        scores = self.get_topscore(have_items, sell_items, options["from_round"], options["to_round"], options["player"], options["multiplier_sell_score"])
        top_scores = sorted(scores.items(), key=lambda x:-x[1])[0: 100]

        self.show_topscore(top_scores)
    
    def add_arguments(self, parser):
        parser.add_argument('--from_round', nargs='?', default=1, type=int)
        parser.add_argument('--to_round', nargs='?', default=18, type=int)
        parser.add_argument('--player', nargs='?', default=[1,2], type=list)
        parser.add_argument('--multiplier_sell_score', nargs='?', default=0.5, type=float)