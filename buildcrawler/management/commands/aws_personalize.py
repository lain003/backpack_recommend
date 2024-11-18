from django.core.management.base import BaseCommand

from buildcrawler.models import Item, Video, Build, BattleSet, Round, Recipe
from buildcrawler.logic.backpack_cv import BackPackCV

import cv2
import os
import numpy as np

import os.path
import mss
import csv
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        battle_sets = BattleSet.objects.all()
        #battle_sets = BattleSet.objects.filter(id=151)
        bought_items = {}
        for battle_set in battle_sets:
            rounds = Round.objects.order_by('num').filter(battle_set=battle_set)
            before_items = []
            bought_items[battle_set.id] = {}
            for round in rounds:
                builds = Build.objects.filter(round=round, player=1)
                mixed_items = []
                for build in builds:
                    items = []
                    for _ in range(build.num):
                        items.append(build.item)
                    mixed_items.extend(Item.get_mixed_items(items))
                
                # diff
                indexs_match_before_item = []
                diff_items = []
                for after_item in mixed_items:
                    is_match = False
                    for i, before_item in enumerate(before_items):
                        if i in indexs_match_before_item:
                            continue
                        if after_item.id == before_item.id:
                            indexs_match_before_item.append(i)
                            is_match = True
                            break
                    if not is_match:
                        diff_items.append(after_item)

                bought_items[battle_set.id][round.id] = diff_items
                
                before_items = mixed_items
            
        with open('ItemInteractions.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["USER_ID", "ITEM_ID", "TIMESTAMP", "ROUND"])
            for battle_set_id, round_items in bought_items.items():
                for round_id, items in round_items.items():
                    ddd = datetime.strptime("2023/1/1","%Y/%m/%d") + timedelta(days=battle_set_id) + timedelta(hours=round_id)
                    unixtime = int(ddd.timestamp())
                    round = Round.objects.get(id=round_id)
                    for item in items:
                        writer.writerow([battle_set_id, item.id, unixtime, round.num])
