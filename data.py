import os
import datetime
import json
from PIL import Image
import collections
import numpy as np
import labelme
from pycocotools.coco import COCO 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
def Read_data(data_file,label_file,):
  now = datetime.datetime.now()
  coco_base = dict(
      info=dict(
          description=None,
          url=None,
          version=None,
          year=now.year,
          contributor=None,
          date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
      ),
      licenses=[dict(url=None, id=0, name=None,)],
      images=[
          # license, url, file_name, height, width, date_captured, id
      ],
      type="instances",
      annotations=[
          # segmentation, area, iscrowd, image_id, bbox, category_id, id
      ],
      categories=[
          # supercategory, id, name
      ],
  )

  list_label=os.listdir(label_file)
  image_id=0
  category_id=1
  annotation_id=0
  imagetoid={}
  categorytoid={}

  for i in list_label:
    filename=i[:i.index('.')]
    k=json.load(open(label_file+'/'+filename+'.json'))

    image_path=k["imagePath"]
    img=Image.open(data_file+'/'+image_path)
    if image_path not in [x['file_name'] for x in coco_base["images"]]:
      imagetoid[image_path]=image_id
      j={
        'license':0,
        'url':None,
        'file_name':image_path,
        'height':img.size[1],
        'width':img.size[0],
        'date_captured':None,
        'id':image_id,
      }
      coco_base["images"].append(j)
      image_id+=1


    shapes=k["shapes"]

    import uuid
    import pycocotools.mask
    masks = {}  # for area
    segmentations = collections.defaultdict(list)  # for segmentation

    for shape in shapes:

      label=shape["label"]

      if label not in [x['name'] for x in coco_base["categories"]]:
        categorytoid[label]=category_id
        j={
            'name':label,
            'id':category_id,
            'supercategory':None
        }
        coco_base["categories"].append(j)
        category_id+=1
        
      points = shape["points"]
      label = shape["label"]
      group_id = shape.get("group_id")
      shape_type = shape.get("shape_type", "polygon")
      mask = labelme.utils.shape_to_mask(
          img.size[::-1], points, shape_type
      )
      if group_id is None:
        group_id = uuid.uuid1()

      instance = (label, group_id)

      if instance in masks:
        masks[instance] = masks[instance] | mask
      else:
        masks[instance] = mask

      if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        points = [x1, y1, x2, y1, x2, y2, x1, y2]
      if shape_type == "circle":
        (x1, y1), (x2, y2) = points
        r = np.linalg.norm([x2 - x1, y2 - y1])
        # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
        # x: tolerance of the gap between the arc and the line segment
        n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
        i = np.arange(n_points_circle)
        x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
        y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
        points = np.stack((x, y), axis=1).flatten().tolist()
      else:
        points = np.asarray(points).flatten().tolist()

      segmentations[instance].append(points)
    segmentations = dict(segmentations)

    for instance, mask in masks.items():
      cls_name, group_id = instance
      if cls_name not in categorytoid:
          continue
      cls_id = categorytoid[cls_name]

      mask = np.asfortranarray(mask.astype(np.uint8))
      mask = pycocotools.mask.encode(mask)
      area = float(pycocotools.mask.area(mask))
      bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
      x={
          'category_id':cls_id,
          'image_id':imagetoid[image_path],
          'id':annotation_id,
          'iscrowd':0,
          'bbox':bbox,
          'area':area,
          'segmentation':segmentations[instance]
      }
      coco_base['annotations'].append(x)
      annotation_id+=1
  return coco_base

tran=transforms.Compose([transforms.ToTensor()])

class detetction_class(Dataset):
  def __init__(self,json_file,datafolder):
    z=json.load(open(json_file))
    self.images=z['images']
    self.annotations=z['annotations']
    self.datafolder=datafolder
    self.coco=COCO(json_file)
  def __getitem__(self, index):
    img=tran(Image.open(os.path.join(self.datafolder, self.images[index]['file_name'])))
    img_id=self.images[index]['id']
    boxes=[]
    masks=[]
    label=[]
    for i in self.annotations:
      if(i['image_id']==img_id):
        
        x1=i['bbox'][0]
        y1=i['bbox'][1]
        x2=x1+i['bbox'][2]
        y2=y1+i['bbox'][3]
        boxes.append(torch.tensor([x1,y1,x2,y2]))
        label.append(i['category_id'])
        masks.append(torch.from_numpy(self.coco.annToMask(i)))

    boxes=torch.stack(boxes)
    masks=torch.stack(masks)
    label=torch.tensor(label,dtype=torch.int64)
    return img,{'boxes':boxes,'labels':label,'masks':masks}

  def __len__(self):     
    return(len(self.images))
    