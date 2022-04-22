from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from torchvision import transforms
from skimage import measure

def visualize(image_v,annotaions,format=0):
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
    bbox = annotaions["boxes"]
    mask=annotaions['masks']
    label=annotaions["labels"]
    for i in range(len(label)):
      if 'scores' in annotaions.keys():
        if annotaions['scores'][i]<0.5:
          continue

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