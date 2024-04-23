from django.core.management.base import BaseCommand

from buildcrawler.models import Item, Video, Build, BattleSet, Round

import itertools
from operator import itemgetter
import cv2

class Command(BaseCommand):
    help = ""

    def hoge(self):
        #テンプレートマッチングでラウンドを特定する方法。精度が不安定なためコメントアウト

        matchs = []
        screen_img = cv2.imread(f'buildcrawler/images/sample/shop.png')
        screen_img_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
        screen_num_img = screen_img_gray[469 : 484, 69 : 87]
        for i in range(1, 18):
            num_img = cv2.imread(f'buildcrawler/images/round_images/{i}.png')
            num_img = cv2.resize(num_img, dsize=(960, 540))
            num_img = num_img[469 : 484, 69 : 87]
            num_img_gray = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)
            match = cv2.matchTemplate(image=screen_num_img, templ=num_img_gray, method=cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(match)
            matchs.append({"value": maxVal, "loc": maxLoc, "num": i})

            im_h = cv2.hconcat([screen_img[469 : 484, 69 : 87], num_img])
            cv2.imshow("After NMS", cv2.resize(im_h,None,fx=2,fy=2))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        import pdb; pdb.set_trace()
        max_match = max(matchs, key=lambda x: x["value"])
        return max_match["num"]

    def handle(self, *args, **options):
        self.hoge()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully closed poll')
        )