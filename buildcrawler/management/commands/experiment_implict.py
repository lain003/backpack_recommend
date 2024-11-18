from django.core.management.base import BaseCommand

import implicit
from implicit.datasets.lastfm import get_lastfm
class Command(BaseCommand):
    help = ""

    @classmethod
    def hoge(cls):
        artists, users, plays = get_lastfm()

        model = implicit.als.AlternatingLeastSquares(factors=50,
                                    regularization=0.01,
                                    iterations=15)
        model.fit(plays)
        user_items = plays.T.tocsr()
        for userid, username in enumerate(users[:5]):
            import pdb; pdb.set_trace()
            artistids, scores =  model.recommend(userid, user_items[userid])
            
        print("a")

    def handle(self, *args, **options):
        Command.hoge()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully closed poll')
        )