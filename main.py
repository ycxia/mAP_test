import os
import json

IOU_THRESH = 0.5

GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')

def line_split(line):
    line = line.strip()
    return line.split()

def IOU(box1, box2):
    # xmin, ymin, xmax, ymax
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1
    inter_area = inter_h * inter_w
    if inter_area <= 0:
        return 0
    union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) - inter_area
    return inter_area / union_area

def voc_ap(reca, prec):
    reca.insert(0, 0.0)
    reca.append(1.0)
    prec.insert(0, 0.0)
    prec.append(0.0)

    maxprec = prec[:]
    for index in range(len(maxprec)-2, -1, -1):
        maxprec[index] = max(maxprec[index], maxprec[index + 1])

    mrec = reca[:]
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i-1]) * maxprec[i]

    return ap

gt_files = os.listdir(GT_PATH)
gt_count_per_class = {}
for gt_file in gt_files:
    gt_bbox = []
    with open(os.path.join(GT_PATH, gt_file), 'r') as f:
        for line in f.readlines():
            class_name, xmin, ymin, xmax, ymax = line_split(line)
            if class_name not in gt_count_per_class.keys():
                gt_count_per_class[class_name] = 1
            else:
                gt_count_per_class[class_name] += 1
            gt_bbox.append({"class_name":class_name, "bbox":[xmin, ymin, xmax, ymax], "used":False})

    with open('tmp_files/'+gt_file.split('.')[0]+'_gr.json', 'w') as f:
        json.dump(gt_bbox, f)

gt_classes = list(gt_count_per_class.keys())
gt_classes.sort()

dr_files = os.listdir(DR_PATH)

for gt_class in gt_classes:
    bounding_box = []
    for dr_file in dr_files:
        f = open(os.path.join(DR_PATH, dr_file))
        for line in f.readlines():
            class_name, conf, xmin, ymin, xmax, ymax = line_split(line)
            if class_name == gt_class:
                bounding_box.append({"confidence":conf, "bbox":[xmin,ymin,xmax,ymax], "file_id":dr_file.split('.')[0]})

    bounding_box.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open('tmp_files/'+ gt_class +'_dr.json','w') as fd:
        json.dump(bounding_box, fd)

#count TP and FP for each class
count_ture_positive = {}

sum_ap = 0.0
foutput = open('output.txt','w')
for gt_class in gt_classes:
    dr_contents = json.load(open('tmp_files/'+ gt_class +'_dr.json','r'))
    dr_class_length = len(dr_contents)
    tp = [0] * dr_class_length
    fp = [0] * dr_class_length
    for index, dr_content in enumerate(dr_contents):
        file_id = dr_content['file_id']
        bbox_dr = [float(x) for x in dr_content['bbox']]
        gt_contents = json.load(open('tmp_files/'+ file_id + '_gr.json', 'r'))
        max_iou = -1
        tmp_matched = -1
        for gt_content in gt_contents:
            if gt_content["class_name"] == gt_class:
                bbox_gt =[float(x) for x in gt_content['bbox']]
                iou = IOU(bbox_dr, bbox_gt)
                if iou > max_iou:
                    max_iou = iou
                    tmp_matched = gt_content


        if max_iou > IOU_THRESH and not tmp_matched['used']:
            tp[index] = 1
            if gt_class not in count_ture_positive.keys():
                count_ture_positive[gt_class] = 1
            else:
                count_ture_positive[gt_class] += 1
            gt_contents[gt_contents.index(gt_content)]["used"] = True
            with open('tmp_files/'+ file_id + '_gr.json', 'w') as f:
                json.dump(gt_contents, f)
        else:
            fp[index] = 1

    #calculate AP for each class
    tmp = 0
    for index, tpval in enumerate(tp):
        tmp += tpval
        tp[index] = tmp
    print(tp)
    tmp=0
    for index, fpval in enumerate(fp):
        tmp += fpval
        fp[index] = tmp

    prec = tp[:]
    reca = tp[:]
    for i in range(len(prec)):
        prec[i] = float(tp[i]) / (tp[i] + fp[i])

    for i in range(len(reca)):
        reca[i] = float(tp[i]) / gt_count_per_class[gt_class]
    foutput.write(gt_class+'\n')
    foutput.write("precison: {}\n".format(prec))
    foutput.write("recall: {}\n".format(reca))

    ap = voc_ap(reca[:], prec[:])
    foutput.write("ap: {}\n".format(ap))
    sum_ap += ap
    print('class: {}  ap:{}\n'.format(gt_class, ap))

print('mAP: {}\n'.format(sum_ap/len(gt_classes)))






