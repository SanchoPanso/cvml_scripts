import numpy as np
import cv2
import os
import colorsys


def main():
    # src_img_dir = r'D:\datasets\geo_ai\geoai_aerials\geiao_aerial_25052023\valid\images'
    # src_lbl_dir = r'D:\datasets\geo_ai\geoai_aerials\geiao_aerial_25052023\valid\predicted_labels'
    # dst_img_dir = r'D:\datasets\geo_ai\geoai_aerials\geiao_aerial_25052023\valid_pred'

    # classes = ['roads', 'Lights pole', 'trees', 'singboard', 'building', 'farms', 'garbage', 'tracks']
    # is_thing = [False, True, True, True, True, True, True, False]
    
    # create_vis(src_img_dir, src_lbl_dir, dst_img_dir, classes, is_thing)
    
    src_img_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_13062023_cropped_1280\train\images'
    src_lbl_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_13062023_cropped_1280\train\labels'
    dst_img_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_13062023_cropped_1280\train_gt'

    
    classes =  ['trees', 'animal farms', 'signboard', 'garbage', 'buildings', 'light poles', 'traffic sign']
    is_thing = [True] * len(classes)
       
    create_vis(src_img_dir, src_lbl_dir, dst_img_dir, classes, is_thing)
    
    # src_img_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_07062023\train\images'
    # src_lbl_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_07062023\train\predicted_labels'
    # dst_img_dir = r'D:\datasets\geo_ai\geoai_360\geoai_360_07062023\train_pred'

    
    # classes =  ['trees', 'animal farms', 'signboard', 'garbage', 'buildings', 'light poles', 'traffic sign']
    # is_thing = [True] * len(classes)
       
    # create_vis(src_img_dir, src_lbl_dir, dst_img_dir, classes, is_thing)


def create_vis(src_img_dir, src_lbl_dir, dst_img_dir, classes, is_thing):
    pallete = get_palette(len(classes))

    os.makedirs(dst_img_dir, exist_ok=True)
    img_files = os.listdir(src_img_dir)

    for img_file in img_files:
        print(img_file)
        img = cv2.imread(os.path.join(src_img_dir, img_file))
        name, ext = os.path.splitext(img_file)
        lbl_path = os.path.join(src_lbl_dir, name + '.txt')
        
        if not os.path.exists(lbl_path):
            cv2.imwrite(os.path.join(dst_img_dir, img_file), img)
            continue
        
        labels = []
        with open(lbl_path) as f:
            text = f.read()
            lines = text.split('\n')
            for line in lines:
                if line.strip() == '':
                    continue
                labels.append(list(map(float, line.split(' '))))
        
        if len(labels) == 0:
            cv2.imwrite(os.path.join(dst_img_dir, img_file), img)
            continue
        
        height, width = img.shape[:2]
        bboxes = []
        pts_list = []
        texts = []
        for lbl in labels:
            cls_id = lbl[0]
            cls_id = int(cls_id)
            
            if len(lbl) % 2 == 0:
                conf = lbl[-1]
                seg_rel = lbl[1:-1]
                xs = (np.array(seg_rel[0::2]) * width).reshape(-1, 1).astype('int32')
                ys = (np.array(seg_rel[1::2]) * height).reshape(-1, 1).astype('int32')
            else:
                conf = None
                seg_rel = lbl[1:]
                xs = (np.array(seg_rel[0::2]) * width).reshape(-1, 1).astype('int32')
                ys = (np.array(seg_rel[1::2]) * height).reshape(-1, 1).astype('int32')
            
            
            x1 = xs.min()
            x2 = xs.max()
            y1 = ys.min()
            y2 = ys.max()
            
            bboxes.append([cls_id, x1, y1, x2, y2])
            
            pts = np.concatenate([xs, ys], axis=1)
            pts = pts.reshape(-1, 1, 2)
            
            pts_list.append(pts)
            
            color = pallete[cls_id]
            mask = np.zeros(img.shape, dtype='uint8')
            mask = cv2.fillPoly(mask, [pts], color)
            
            img = add_color_mask(img, mask)
            
            class_name = classes[cls_id]
            if conf is None:
                text = f'{class_name}'
            else:
                conf = '{0:.3f}'.format(conf)
                text = f'{class_name} {conf}'
            
            texts.append(text)
        
        for i in range(len(bboxes)):
            cls_id, x1, y1, x2, y2 = bboxes[i]
            text = texts[i]
            
            color = pallete[cls_id]
            
            if is_thing[cls_id]:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 + text_height), color, -1)
                cv2.putText(img, text=text, org=(x1, y1 + int(0.8 * text_height)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),
                                lineType=cv2.LINE_AA)
            else:
                pts = pts_list[i]
                ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                for pt in pts:
                    x, y = pt[0][0], pt[0][1]
                    if 0 <= x < img.shape[1] - text_width and 0 <= y < img.shape[0] - text_height:
                        break
                
                cv2.rectangle(img, (x, y), (x + text_width, y + text_height), color, -1)
                cv2.putText(img, text=text, org=(x, y + int(0.8 * text_height)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),
                                lineType=cv2.LINE_AA)
            
        # cv2.imshow('img', cv2.resize(img, (600, 600)))
        # cv2.waitKey()
        cv2.imwrite(os.path.join(dst_img_dir, img_file), img)


def get_palette(num_of_colors: int) -> list:
    hsv_tuples = [((x / num_of_colors) % 1, 0.9, 0.7) for x in range(num_of_colors)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    rgb_tuples = list(map(lambda x: (int(x[1] * 255), int(x[0] * 255), int(x[2] * 255)), rgb_tuples))
    return rgb_tuples

def add_color_mask(img, mask, alpha=0.5):
    ret, binary_mask = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 1, 1, cv2.THRESH_BINARY)

    binary_mask = binary_mask.astype('float32') * alpha
    binary_mask = cv2.merge([binary_mask] * 3)
    
    img = img.astype('float32')
    mask = mask.astype('float32')
    
    res = np.clip((img * (1 - binary_mask) + mask * binary_mask), 0, 255).astype('uint8')
    
    return res


if __name__ == '__main__':
    main()


        
        
        
    
    

