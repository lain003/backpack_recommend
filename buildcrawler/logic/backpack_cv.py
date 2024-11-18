import cv2
import numpy as np
from buildcrawler.models import Item, Video, Build, BattleSet, Round

class BoxAndScore:
    def __init__(self, box: list[int], score: int):
        self.box = box
        self.score = score
    
    def __str__(self):
        return f"box = {self.box}, score = {self.score}"

class ItemMatchResult:
    def __init__(self, item: Item, box_and_scores: list[BoxAndScore]):
        self.item = item
        self.box_and_scores= box_and_scores

    def __str__(self):
        return f"item = {self.item}, box_and_scores = {self.box_and_scores}"

class BackPackCV:
    
    @classmethod
    def rotate_match_template(cls, screen_img, item, color_screen_img = None) -> ItemMatchResult:
        temp_img_path = f'buildcrawler/images/items/{item.name}_25.png'
        thresh = item.threshold
        temp_img = cv2.imread(temp_img_path)
        temp_gray_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

        _, template_mask_img = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY)
        template_mask_img = cv2.cvtColor(template_mask_img, cv2.COLOR_BGR2GRAY)
        H, W = temp_img.shape[:2]
        boxes = np.empty((0,4), int)
        boxe_and_scores = []
        
        for i in range(4):
            match = cv2.matchTemplate(image=screen_img, templ=temp_gray_img, method=cv2.TM_CCOEFF_NORMED, mask=template_mask_img)
            
            # np.unravel_index(np.argmax(match), match.shape) #最大値
            (y_points, x_points) = np.where(match >= thresh)
            
            if 0 != len(x_points):
                tmp_boxes = list()
                for (x, y) in zip(x_points, y_points):
                    if i % 2 == 1:
                        tmp_boxe = (x, y, x + H, y + W)
                    else:
                        tmp_boxe = (x, y, x + W, y + H)

                    # # 色を比較するコード。
                    # screen_img_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(color_screen_img[tmp_boxe[1] : tmp_boxe[3], tmp_boxe[0] : tmp_boxe[2]]))
                    # temp_img_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(temp_img))
                    # delta_E = colour.delta_E(screen_img_lab, temp_img_lab)
                    # print(np.mean(delta_E))
                    # print(np.median(delta_E))

                    tmp_boxes.append(tmp_boxe)
                
                boxes = np.append(boxes, tmp_boxes, axis=0)
                for boxe in tmp_boxes:
                    boxe_and_scores.append(BoxAndScore(boxe, match[boxe[1]][boxe[0]]))
                #_, maxVal, _, _ = cv2.minMaxLoc(match)
            temp_gray_img = cv2.rotate(temp_gray_img, cv2.ROTATE_90_CLOCKWISE)
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)
            template_mask_img = cv2.rotate(template_mask_img, cv2.ROTATE_90_CLOCKWISE)

        return ItemMatchResult(item, boxe_and_scores)
    
    @classmethod
    def check_duplicate_box(_, source_box, target_box, threshold_width, threshold_height) -> bool:
        if not (source_box[0] - threshold_width <= target_box[0] and target_box[0] <= source_box[0] + threshold_width):
            return False
        if not (source_box[1] - threshold_height <= target_box[1] and target_box[1] <= source_box[1] + threshold_height):
            return False
        if not (source_box[2] - threshold_width <= target_box[2] and target_box[2] <= source_box[2] + threshold_width):
            return False
        if not (source_box[3] - threshold_height <= target_box[3] and target_box[3] <= source_box[3] + threshold_height):
            return False
        return True
    
    @classmethod
    def del_duplicate_box(_, item_match_results: list[ItemMatchResult], thresh=0.1) -> list[ItemMatchResult]:
        result_indexs = {} #{0:[0,1,2], 1:[0,1]}
        for i in range(0, len(item_match_results)):
            result_indexs[i] = []
            for j in range(0, len(item_match_results[i].box_and_scores)):
                result_indexs[i].append(j)
        
        del_keys = [] # [(11, 2), (11, 3)...
        
        for result_index in result_indexs.keys():
            item_match_result = item_match_results[result_index]
            for score_box_index in result_indexs[result_index]:
                if (result_index, score_box_index) in del_keys:
                    continue
                box_and_score = item_match_result.box_and_scores[score_box_index]
                score_box = box_and_score.box
                threshold_width = (score_box[2] - score_box[0]) * thresh
                threshold_height = (score_box[3] - score_box[1]) * thresh
                for result_index2 in result_indexs.keys():
                    for score_box_index2 in result_indexs[result_index2]:
                        if (result_index2 == result_index and score_box_index2 == score_box_index) or ((result_index2,score_box_index2) in del_keys):
                            continue
                        box_and_score2 = item_match_results[result_index2].box_and_scores[score_box_index2]
                        if BackPackCV.check_duplicate_box(score_box, box_and_score2.box, threshold_width, threshold_height):
                            if box_and_score.score > box_and_score2.score:
                                # if item_match_result.item.name != item_match_results[result_index2].item.name:
                                #     print(f"duplicate result {item_match_result.item} {box_and_score.score} > {item_match_results[result_index2].item} {box_and_score2.score}")
                                del_keys.append((result_index2,score_box_index2))
                            else:
                                # if item_match_result.item.name != item_match_results[result_index2].item.name:
                                #     print(f"duplicate result {item_match_result.item} {box_and_score.score} < {item_match_results[result_index2].item} {box_and_score2.score}")
                                del_keys.append((result_index,score_box_index))
                                continue

        return_item_match_results = []
        for i, item_match_result in enumerate(item_match_results):
            return_item_match_result = ItemMatchResult(item_match_result.item, [])
            for j, box_and_score in enumerate(item_match_result.box_and_scores):
                if not (i,j) in del_keys:
                    return_item_match_result.box_and_scores.append(box_and_score)
            if len(return_item_match_result.box_and_scores) != 0:
                return_item_match_results.append(return_item_match_result)

        return return_item_match_results
    
    @classmethod
    def outputimg_results(cls, img, item_match_results, identifer="None"):
        for item_match_result in item_match_results:
            print(f"{item_match_result.item} {len(item_match_result.box_and_scores)}")
            # Build.objects.create(item=item_match_result.item, num=len(item_match_result.box_and_scores), player=2)
            for box_and_score in item_match_result.box_and_scores:
                box = box_and_score.box
                cv2.putText(img,item_match_result.item.name, ( box[0], box[1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
                cv2.rectangle(img, (box[0], box[1] + 5), (box[2], box[3]), (255, 0, 0), 3)
        # cv2.imwrite(f'buildcrawler/images/result/{identifer}.png', img)
        cv2.imshow("After NMS", cv2.resize(img,None,fx=2,fy=2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def identify_items(cls, img, debug=False) -> list[ItemMatchResult]:
        #import pdb; pdb.set_trace()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        items = Item.objects.all()
        item_match_results = []
        for item in items:
            item_match_result = BackPackCV.rotate_match_template(img_gray, item, color_screen_img=img)
            
            if len(item_match_result.box_and_scores) >= 1:
                item_match_results.append(item_match_result)
        
        item_match_results = BackPackCV.del_duplicate_box(item_match_results, thresh=0.2)
        if debug:
            BackPackCV.outputimg_results(img, item_match_results)
        return item_match_results