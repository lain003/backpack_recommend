from django.core.management.base import BaseCommand, CommandError

import cv2
import numpy as np

from imutils.object_detection import non_max_suppression

from buildcrawler.models import Item, Video, Build, BattleSet, Round
from buildcrawler.logic.backpack_cv import BackPackCV, ItemMatchResult
from operator import itemgetter

import colour
class Command(BaseCommand):
    help = ""

    @classmethod
    def outputimg_results(cls, img, item_match_results, identifer):
        for item_match_result in item_match_results:
            print(f"{item_match_result.item} {len(item_match_result.box_and_scores)}")
            # Build.objects.create(item=item_match_result.item, num=len(item_match_result.box_and_scores), player=2)
            for box_and_score in item_match_result.box_and_scores:
                box = box_and_score.box
                cv2.putText(img,item_match_result.item.name, ( box[0], box[1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
                cv2.rectangle(img, (box[0], box[1] + 5), (box[2], box[3]), (255, 0, 0), 3)
        cv2.imwrite(f'buildcrawler/images/result/{identifer}.png', img)
        # cv2.imshow("After NMS", cv2.resize(img,None,fx=2,fy=2))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    @classmethod
    def save_builds(cls, item_match_results: list[ItemMatchResult], round: Round, player: int):
        for item_match_result in item_match_results:
            Build.objects.create(round=round, item=item_match_result.item, player=player, num=len(item_match_result.box_and_scores))
    
    @classmethod
    def show_results(cls, img, item_match_results):
        for item_match_result in item_match_results:
            print(f"{item_match_result.item} {len(item_match_result.box_and_scores)}")
            # Build.objects.create(item=item_match_result.item, num=len(item_match_result.box_and_scores), player=2)
            for box_and_score in item_match_result.box_and_scores:
                box = box_and_score.box
                cv2.putText(img,item_match_result.item.name, ( box[0], box[1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
                cv2.rectangle(img, (box[0], box[1] + 5), (box[2], box[3]), (255, 0, 0), 3)
        cv2.imshow("After NMS", cv2.resize(img,None,fx=2,fy=2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def default(cls, video_id: int):
        video = Video.objects.get(pk=video_id)
        battle_sets = BattleSet.objects.filter(video=video)
        for battle_set in battle_sets:
            rounds = Round.objects.filter(battle_set=battle_set)
            for round in rounds:
                print(f"round = {round.id}")
                img = cv2.imread(f'buildcrawler/images/screenshots/{video.id}/{round.id}.png')
                H, W = img.shape[:2]
                top_img = img[0 : int(H * 0.6), 0 : W]
                top_H, _ = top_img.shape[:2]
                player1_img = top_img[0 : top_H, 0 : int(W / 2)]
                player2_img = top_img[0 : top_H, int(W / 2) : W]
                item_match_results_1 = BackPackCV.identify_items(player1_img)
                item_match_results_2 = BackPackCV.identify_items(player2_img)
                Command.outputimg_results(player1_img, item_match_results_1, f"{round.id}_1")
                Command.outputimg_results(player2_img, item_match_results_2, f"{round.id}_2")
                Command.save_builds(item_match_results_1, round, 1)
                Command.save_builds(item_match_results_2, round, 2)
    
    @classmethod
    def experience(cls):
        img = cv2.imread(f'buildcrawler/images/screenshots/{10}/{273}.png')
        H, W = img.shape[:2]
        top_img = img[0 : int(H * 0.6), 0 : W]
        top_H, _ = top_img.shape[:2]
        player1_img = top_img[0 : top_H, 0 : int(W / 2)]
        player2_img = top_img[0 : top_H, int(W / 2) : W]
        Command.show_results(player2_img, BackPackCV.identify_items(player2_img))
        Command.show_results(player1_img, BackPackCV.identify_items(player1_img))

    @classmethod
    def re_build(cls, video_id: int):
        battle_sets = BattleSet.objects.select_related("round").filter(video=video_id)
        rounds = Round.objects.filter(battle_set__in=battle_sets)
        Build.objects.filter(round__in=rounds).delete()

        cls.default(video_id=video_id)

    def handle(self, *args, **options):
        video_id = options["video_id"]
        if options["type"] == "":
            Command.default(video_id=video_id)
        elif options["type"] == "re_build":
            Command.re_build(video_id=video_id)
        else:
            Command.experience()
    
    def add_arguments(self, parser):
        parser.add_argument('--type', nargs='?', default='', type=str)
        parser.add_argument('--video_id', nargs='?', default='', type=int)