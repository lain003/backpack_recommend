from django.core.management.base import BaseCommand

from buildcrawler.models import Item, Video, Build, BattleSet, Round

import itertools
from operator import itemgetter
import cv2
import boto3
import datetime

class Command(BaseCommand):
    help = ""

    @classmethod
    def hoge(cls):
        # personalize = boto3.client('personalize')
        # response = personalize.create_event_tracker(
        #     name='MyEventTracker',
        #     datasetGroupArn='arn:aws:personalize:ap-northeast-1:620988379686:dataset-group/bpb_datasetgroup'
        # )
        # print(response["trackingId"])
        # trackingId = response["trackingId"]
        trackingId = "84770292-e4c5-4b1d-b266-d1ff45f8a295"

        # 26	0.0966112	-
        # 2	0.0917398	-
        # 27	0.0743260	-
        # 1	0.0639666	-
        # 25	0.0581408
        # 24	0.0546955


        # 27	0.1032990	-
        # 2	0.0759279	-
        # 24	0.0657276	-
        # 25	0.0654240	-
        # 21	0.0620784	-
        # 1	0.0585747

        client = boto3.client('personalize-events')
        response = client.put_events(
            trackingId=trackingId,
            userId='10001',
            sessionId='aaa1',
            eventList=[
                {
                    'eventType': 'string',
                    'sentAt': 1716658547,
                    "itemId": "21",
                    "properties": '{"round":"5"}'
                },
            ]
        )


    @classmethod
    def fuga(cls):
        client = boto3.client('personalize-runtime')
        response = client.get_recommendations(
            campaignArn="arn:aws:personalize:ap-northeast-1:620988379686:campaign/mycampaign",
            userId="10001",
            context={"round": "16"}
        )

        print(response["itemList"])

    def add_arguments(self, parser):
        parser.add_argument('--type', nargs='?', default='', type=str)

    def handle(self, *args, **options):
        if options["type"] == "1":
            Command.hoge()
        else:
            Command.fuga()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully closed poll')
        )