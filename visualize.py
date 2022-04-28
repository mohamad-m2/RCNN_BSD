from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from torchvision import transforms
from skimage import measure
import torch
def visualize(image_v,annotations,format=0):
  if(format==0):
    image_path = 'BSData/data/'+image_v["file_name"]
    image = cv2.imread(image_path)
    for i, ann in enumerate(annotations):
      segs = ann["segmentation"]
      bbox = np.array(ann["bbox"])
      bbox[2:4] = bbox[0:2] + bbox[2:4]
      segs = [np.array(seg, np.int32).reshape((1, -1, 2))
              for seg in segs]
      
      for seg in segs: 
        cv2.drawContours(image, seg, -1, (50,200,100), 3)# image, point, idk why -1, color,thickness

      cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), (100,50,200), 3)
    cv2_imshow(image)

  else:
    image=np.array(transforms.ToPILImage()(image_v)) 

    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    if("scores" in annotations):
      annotations=post_process(annotations)
    bbox = annotations["boxes"]
    mask=annotations['masks']
    label=annotations["labels"]

    for i in range(len(label)):

      if label[i].item()==0:
        continue
      else:
        print('box drawing')
        cv2.rectangle(image, (int(bbox[i][0]), int(bbox[i][1])), 
                    (int(bbox[i][2]), int(bbox[i][3])), (100,50,200), 3)
        

        segs=[]
        ok=mask[i].cpu().detach().numpy()
        if(len(ok.shape)>2):
          ok=ok[0]
        contours = measure.find_contours(ok, 0.5)
        for contour in contours:
          contour = np.flip(contour, axis=1)
          segmentation = contour.ravel().tolist()
          segs.append(segmentation)
        segs = [np.array(seg, np.int32).reshape((1, -1, 2))
              for seg in segs]
      
        for seg in segs: 
          cv2.drawContours(image, seg, -1, (50,200,100), 1)
        
    cv2_imshow(image)


def get_intersection(bb1, bb2):
  """
  Calculate the Intersection over Union (IoU) of two bounding boxes.

  Parameters
  ----------
  bb1 : tensor
      [x1,y1,x2,y2]
  bb2 : tensor
      [x1,y1,x2,y2]

  Returns
  -------
  float
      in [0, 1]
  """

  # determine the coordinates of the intersection rectangle
  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
  bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
  intersection = intersection_area / min(bb1_area,bb2_area)
  return intersection

def post_process(annotation,intersection_threshold=0.5,score_threshold=0.25):
  mask=[]
  scores=[]
  label=[]
  box=[]
  j=0
  indexes=[1]*annotation["scores"].shape[0]
  while(j<annotation["scores"].shape[0]):
    if(indexes[j]==1):
      for i in range(j+1,annotation["scores"].shape[0]):
        intersection=get_intersection(annotation["boxes"][i],annotation["boxes"][j])
        if(intersection>intersection_threshold):
          indexes[i]=0
    j+=1
  
  for i in range(len(indexes)):
    if(indexes[i]==1 and annotation["scores"][i]>score_threshold):
        mask.append(annotation["masks"][i])
        scores.append(annotation["scores"][i])
        label.append(annotation["labels"][i])
        box.append(annotation["boxes"][i])
  if(len(scores)>0):
    return {
      "scores":torch.stack(scores),
      "masks":torch.stack(mask),
      "boxes":torch.stack(box),
      "labels":torch.stack(label),
    }
  else:
    return{
      "scores":torch.tensor([]),
      "masks":torch.tensor([]),
      "boxes":torch.tensor([]),
      "labels":torch.tensor([]),
    }




