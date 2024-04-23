from django.core.management.base import BaseCommand, CommandError
from buildcrawler.models import Video, Build, BattleSet, Round
from buildcrawler.logic.const import Const
import abc
import datetime
import time

import cv2 
import numpy as np 
from imutils.object_detection import non_max_suppression 


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep

import pyocr
import os
import re
from PIL import Image

import boto3

from logging import getLogger

logger = getLogger(__name__)

# Round.objects.filter(battle_set__in=[9,8]).delete()
# BattleSet.objects.filter(pk__in=[9,8]).delete()

class NotMatchRank(Exception):
    def __str__(self):
        return (f"NotMatchRank")

class NotFoundRoundImage(Exception):
    def __str__(self):
        return (f"NotFoundRoundImage")

class MissMatchRoundOCR(Exception):
    def __str__(self):
        return (f"MissMatchRoundOCR") 

class BaseState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self):
        pass

class WaitBatlleState():
    def __init__(self, video, round, battle_set):
        self.video = video
        self.round = round
        self.battle_set = battle_set

    def run(self, screen_img) -> BaseState:
        screen_img_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
        temp_img = cv2.imread(f'buildcrawler/images/parts/battle_icon_25.png')
        temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        match = cv2.matchTemplate(screen_img_gray, templ=temp_img_gray, method=cv2.TM_CCOEFF_NORMED)
        (y_points, x_points) = np.where(match >= 0.8)
        if len(x_points) != 0:
            try:
                os.mkdir(f'buildcrawler/images/screenshots/{self.video.id}')
            except FileExistsError:
                pass
            cv2.imwrite(f'buildcrawler/images/screenshots/{self.video.id}/{self.round.id}.png', screen_img)
            return WaitShopState(video=self.video, battle_set=self.battle_set)
        else:
            return WaitBatlleState(video=self.video, round=self.round, battle_set=self.battle_set)

class WaitShopState(BaseState):
    def __init__(self, video, battle_set):
        self.video = video
        self.battle_set = battle_set
    
    def get_rank(self, screen_img_gray) -> str:
        matchs = []
        for rank in ["platinum", "diamond", "master"]:
            temp_img = cv2.imread(f'buildcrawler/images/ranks/{rank}_25.png')
            temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            match = cv2.matchTemplate(image=screen_img_gray, templ=temp_img_gray, method=cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(match)
            matchs.append({"value": maxVal, "loc": maxLoc, "rank": Const.RANKS[rank]})
        max_match = max(matchs, key=lambda x: x["value"])
        return max_match["rank"]
    
    def get_round(self, screen_img, screen_img_gray) -> int:
        temp_img = cv2.imread('buildcrawler/images/parts/clock_25.png')
        temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        H, W = temp_img.shape[:2]
        match = cv2.matchTemplate(image=screen_img_gray, templ=temp_img_gray, method=cv2.TM_CCOEFF_NORMED)
        (y_points, x_points) = np.where(match >= 0.7)
        if len(x_points) == 0:
            raise NotFoundRoundImage
        
        tmp_boxes = list()
        for (x, y) in zip(x_points, y_points):
            tmp_boxes.append((x, y, x + W, y + H))
        tmp_boxes = non_max_suppression(np.array(tmp_boxes))

        x1, y1, _, _ = tmp_boxes[0]
        #img2 = screen_img[y1 + 60: y1 + 140, x1 - 75 : x1 - 5]
        img2 = screen_img[y1 + 15: y1 + 35, x1 - 19 : x1 - 2]
        cv2.imwrite('round.png', img2)

        textract = boto3.client("textract", region_name="ap-southeast-1")

        # read image to bytes
        with open('round.png', 'rb') as f:
            data = f.read()

        # Call Amazon Textract
        response = textract.detect_document_text(
            Document={'Bytes': data}
        )

        lines = filter(lambda x: x["BlockType"] == "LINE", response["Blocks"])
        sorted_lines = sorted(lines, key=lambda x: x['Confidence'], reverse=True)
        if len(sorted_lines) == 0:
            raise MissMatchRoundOCR
            
        try:
            num = int(re.sub(r"\D", "", sorted_lines[0]["Text"]))
        except ValueError:
            raise MissMatchRoundOCR
        if num == 0:
            raise MissMatchRoundOCR
        return num
        

        # #テンプレートマッチングでラウンドを特定する方法。精度が不安定なためコメントアウト
        # matchs = []
        # screen_num_img = screen_img_gray[469 : 484, 69 : 87]
        # for i in range(1, 18):
        #     num_img = cv2.imread(f'buildcrawler/images/round_images/{i}.png')
        #     num_img = cv2.resize(num_img, dsize=(960, 540))
        #     num_img = num_img[469 : 484, 69 : 87]
        #     num_img_gray = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)
        #     match = cv2.matchTemplate(image=screen_num_img, templ=num_img_gray, method=cv2.TM_CCOEFF_NORMED)
        #     _, maxVal, _, maxLoc = cv2.minMaxLoc(match)
        #     matchs.append({"value": maxVal, "loc": maxLoc, "num": i})
        # max_match = max(matchs, key=lambda x: x["value"])
        # return max_match["num"]

    def get_rank_round(self, screen_img):
        screen_img_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
        round = self.get_round(screen_img, screen_img_gray)
        print(f'Round = {round}')

        rank = self.get_rank(screen_img_gray)
        print(f'Rank = {rank}')

        return rank, round
    
    def run(self, screen_img) -> BaseState:
        screen_img_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)

        temp_img = cv2.imread('buildcrawler/images/parts/flower_25.png')
        temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(image=screen_img_gray, templ=temp_img_gray, method=cv2.TM_CCOEFF_NORMED)
        (y_points, x_points) = np.where(match >= 0.6)
        if len(x_points) != 0:
            rank, now_round = self.get_rank_round(screen_img)

            if self.battle_set == None:
                self.battle_set = BattleSet.objects.create(video=self.video, rank=rank)
            else:
                last_round = Round.objects.filter(battle_set=self.battle_set).order_by('-num').first()
                if now_round < last_round.num:
                    self.battle_set = BattleSet.objects.create(video=self.video, rank=rank)
            round = Round.objects.create(num=now_round, battle_set=self.battle_set)
            
            return WaitBatlleState(video=self.video, round=round, battle_set=self.battle_set)
        else:
            return WaitShopState(video=self.video, battle_set=self.battle_set)

class Command(BaseCommand):
    help = ""

    @classmethod
    def rotate_match_template(cls, screen_img, temp_img_path, thresh):
        temp_img = cv2.imread(temp_img_path)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        temp_img = cv2.Laplacian(temp_img, cv2.CV_32F)
        boxes = np.empty((0,4), int)
        for x in range(3):
            H, W = temp_img.shape[:2]
            match = cv2.matchTemplate(
            image=screen_img, templ=temp_img,
            method=cv2.TM_CCOEFF_NORMED)
            (y_points, x_points) = np.where(match >= thresh)
            if len(x_points) != 0:
                tmp_boxes = list() 
                for (x, y) in zip(x_points, y_points): 
                    tmp_boxes.append((x, y, x + W, y + H))
                tmp_boxes = non_max_suppression(np.array(tmp_boxes))
                boxes = np.append(boxes, tmp_boxes, axis=0)
            
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)
        return boxes
    
    @classmethod
    def get_twich_date(cls, date_s):
        now_datetime = datetime.datetime.now()

        if '時間前' in date_s:
            num = int(re.sub(r"\D", "", date_s))
            video_date = now_datetime - datetime.timedelta(hours=-num)
        elif '昨日' in date_s:
            video_date = now_datetime - datetime.timedelta(days=-1)
        elif '一昨日' in date_s:
            video_date = now_datetime - datetime.timedelta(days=-2)
        elif '日前' in date_s:
            num = int(re.sub(r"\D", "", date_s))
            video_date = now_datetime - datetime.timedelta(days=-num)
        elif '先月' in date_s:
            video_date = now_datetime - datetime.timedelta(days=-30)
        elif 'か月前' in date_s:
            num = int(re.sub(r"\D", "", date_s))
            video_date = now_datetime - datetime.timedelta(days=-30*num)
        else:
            raise RuntimeError("Not Match video_date")
        
        return video_date

    @classmethod
    def hoge(cls):
        options = Options()
        options.add_experimental_option("excludeSwitches", ['enable-automation'])
        options.add_argument('--headless=new')
        options.add_argument("--window-size=960,540")
        options.add_argument("--start-fullscreen")
        options.add_argument("--mute-audio")
        driver = webdriver.Chrome(options=options)
        #2076657518, 2077899070, 2074495008, 2056783478, 2063672430, 2063094795, 2066369500, 2063484024, 2068125135, 2068448340, 2069746822, 2081409064
        #2067159288, 2068170869, 2069091488, 2071030419, 2073850488, 2074735050, 2075053779, 2075702215, 2076027882, 2076700482, 2076962327, 2077605460
        #2078278875, 2080778407, 2081250180, 2069564249, 2073279615, 2070404943, 2071311361, 2073399594, 2075233819, 2079097642, 2080152442, 2081980415
        #2077306546, 2082148338
        twich_video_id = 2077306546
        twich_url = f"https://www.twitch.tv/videos/{twich_video_id}"
        driver.get(twich_url)
        
        #sleep(20)
        #print("広告まち終わり")
        
        
        # 最高品質の画質
        try:
            driver.find_element(By.CSS_SELECTOR,"[aria-label='設定']").click()
            driver.find_element(By.CSS_SELECTOR,"[data-a-target='player-settings-menu-item-quality']").click()
            driver.find_element(By.CSS_SELECTOR,"[data-a-target='player-settings-menu']").find_elements(By.CSS_SELECTOR, "[role='menuitemradio']")[1].click()
        
            driver.find_element(By.CSS_SELECTOR,"[aria-label='フルスクリーン（f）']").click()
            date_s = driver.find_element(By.CLASS_NAME, "timestamp-metadata__bar").find_element(By.XPATH, "../p").text
            date = Command.get_twich_date(date_s)
        except:
            driver.get_screenshot_as_file("./start_error.png")
            return
        
        try:
            video = Video.objects.get(twich_video_id=twich_video_id)
            print("既に登録したビデオです")
            return
        except Video.DoesNotExist:
            video = Video(twich_video_id=twich_video_id, time=date, language = 2)
            video.save()

        state = WaitShopState(video=video, battle_set=None)
        while True:
            try:
                if driver.current_url != twich_url: #動画が終わったら次の動画に自動的に遷移するため
                    break
                driver.get_screenshot_as_file("./tmp.png")
                image = Image.open('./tmp.png')
                new_image = image.resize((960, 540))
                new_image.save('./tmp.png')
                state = state.run(cv2.imread('tmp.png'))
                print(state)
            except (NotMatchRank, NotFoundRoundImage, MissMatchRoundOCR) as e:
                print(e)
                sleep(10)
        driver.quit()

    def handle(self, *args, **options):
        Command.hoge()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully closed poll')
        )





