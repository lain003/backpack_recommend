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

    def handle(self, *args, **options):
        def concat_tile(im_list_2d):
            return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

        round_id = options["round_id"]
        round = Round.objects.get(id=round_id)
        set_rounds = Round.objects.filter(battle_set_id=round.battle_set_id)
        
        images_1 = []
        images_2 = []
        for i, round in enumerate(set_rounds):
            img = cv2.imread(f'buildcrawler/images/result/{round.id}_1.png')
            cv2.putText(img, f"{round.num}", (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3, cv2.LINE_AA)
            images_2.append(img)
            if (i + 1) % 5 == 0:
                images_1.append(images_2)
                images_2 = []

        image_2_len = len(images_2)
        if image_2_len != 0:
            if image_2_len != 5:
                for i in range(5-image_2_len):
                    images_2.append(img)
            images_1.append(images_2)

        im_tile = concat_tile(images_1)
        
        cv2.imshow("After NMS", cv2.resize(im_tile,None,fx=0.7,fy=0.7))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def add_arguments(self, parser):
        parser.add_argument('--round_id', nargs='?', default=1, type=int)