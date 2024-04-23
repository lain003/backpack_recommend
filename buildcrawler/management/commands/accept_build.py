from django.core.management.base import BaseCommand

from buildcrawler.models import Item, Video, Build, BattleSet, Round

class Command(BaseCommand):
    help = ""

    @classmethod
    def handle(self, *args, **options):
        # f'{round_id}_{player}'
        rejects = ["6_1","14_2","18_1","18_2","41_2","44_1","44_2"]
        round_range = range(1, 60 + 1)
        for round_id in round_range:
            for player in [1,2]:
                builds = Build.objects.filter(round=round_id, player=player)
                if f'{round_id}_{player}' in  rejects:
                    for build in builds:
                        build.accept = False
                        build.save()
                else:
                    for build in builds:
                        build.accept = True
                        build.save()
                
        