from __future__ import annotations
from django.db import models


# Create your models here.
class Item(models.Model):
    name = models.CharField(max_length=100, blank=False)
    threshold = models.FloatField(default=0.6, blank=False)
    gold = models.IntegerField(blank=False, db_index=True, default=0)

    def __str__(self):
        return self.name
    
    @classmethod
    def get_evolve_item(cls, item) -> list[Item]:
        recipes = Recipe.objects.filter(mixed_item_id=item)
        if len(recipes) == 0:
            return []

        return_items = []
        for recipe in recipes:
            items = cls.get_evolve_item(recipe.result_item)
            if len(items) != 0:
                return_items.append(recipe.result_item)
                return_items.extend(items)
            else:
                return_items.append(recipe.result_item)
        # WoodenSwordの場合はFalconBladeの関係で重複する
        return list(dict.fromkeys(return_items))
    
    @classmethod
    def get_evolve_items(cls, items: list[Item]) -> list[Item]:
        return_items = []
        for item in items:
            mixed_items = cls.get_evolve_item(item)
            if len(mixed_items) == 0:
                return_items.append(item)
            else:
                return_items.extend(mixed_items)
        return return_items
    
    @classmethod
    def get_mixed_item(cls,item) -> list[Item]:
        recipes = Recipe.objects.filter(result_item=item)
        if len(recipes) == 0:
            return []
        
        return_items = []
        for recipe in recipes:
            for _ in range(recipe.num):
                items = cls.get_mixed_item(recipe.mixed_item)
                if len(items) != 0:
                    return_items.extend(items)
                else:
                    return_items.append(recipe.mixed_item)
        return return_items
    
    @classmethod
    def get_mixed_items(cls,items: list[Item]) -> list[Item]:
        return_items = []
        for item in items:
            mixed_items = cls.get_mixed_item(item)
            if len(mixed_items) == 0:
                return_items.append(item)
            else:
                return_items.extend(mixed_items)
        return return_items

class Video(models.Model):
    twich_video_id = models.IntegerField(blank=False, unique=True, db_index=True)
    time = models.DateTimeField(blank=False, db_index=True)
    language = models.IntegerField(blank=False, db_index=True, help_text='1 = en, 2 = jp')

    def __str__(self):
        return str(self.twich_video_id)

class Recipe(models.Model):
    result_item = models.ForeignKey(Item, on_delete=models.PROTECT, related_name='result_item')
    mixed_item = models.ForeignKey(Item, on_delete=models.PROTECT, related_name='mixed_item')
    num = models.IntegerField(blank=False)

class BattleSet(models.Model):
    video = models.ForeignKey(Video, on_delete=models.PROTECT)
    rank = models.IntegerField(blank=False, db_index=True)

class Round(models.Model):
    num = models.IntegerField(blank=False)
    battle_set = models.ForeignKey(BattleSet, on_delete=models.PROTECT)
    result = models.IntegerField(db_index=True, help_text='不明ならnull,勝ちなら1,負けなら2', null=True)

class Build(models.Model):
    round = models.ForeignKey(Round, on_delete=models.PROTECT)
    item = models.ForeignKey(Item, on_delete=models.PROTECT)
    num = models.IntegerField(blank=False)
    player = models.IntegerField(db_index=True, help_text='不明ならnull,配信者なら1,対戦相手なら2', null=True)
    accept = models.BooleanField(null=True, help_text='初期値がnull, 認証してたらtrue, rejectならfalse')

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["round", "item", "player"],
                name="build_round_item_player"
            ),
        ]