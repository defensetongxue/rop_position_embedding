import json
import os
import torch
from VesselSegModule import generate_vessel_result
from utils_ import generate_diffusion_heatmap,generate_position_map
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
def generate_ridge_diffusion(data_path,):
    os.makedirs(os.path.join(data_path,'ridge_diffusion'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_diffusion')}/*")
    splits=['train','val','test']
    for split in splits:
        with open(os.path.join(data_path,'ridge',f'{split}.json'),'r') as f:
            data_list=json.load(f)
        new_data_list=[]
        for data in data_list:
            mask = generate_diffusion_heatmap(data['image_path'],data['ridge_coordinate'], factor=0.5, Gauss=False)
            mask_save_name=data['image_name']/split('.')[0]+'.png'
            mask_save_path=os.path.join(data_path,'ridge_diffusion',mask_save_name)
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_save_path)
            data['diffusion_mask_path']=mask_save_path
            new_data_list.append(data)
        with open(os.path.join(data_path,'ridge',f'{split}.json'),'w') as f:
            json.dump(new_data_list,f)

def generate_possion_map(data_path, compress_r):
    # Generate path_image folder
    os.makedirs(os.path.join(data_path,'position_map_gt'),exist_ok=True)
    
    os.system(f"rm -rf {os.path.join(data_path,'position_map_gt')}/*")

    splits=['train','val']
    for split in splits:
        with open(os.path.join(data_path,'ridge',f'{split}.json'),'r') as f:
            data_list=json.load(f)

        annotate=[]
        for  data in data_list:
            mask = Image.open(data['diffusion_mask_path'])
            mask[mask!=0]=1
            position_save_path=os.path.join(data_path,'position_map_gt',data['image_name'])
            pos_heatmap=generate_position_map(mask,compress_r,save_path=position_save_path)
            # from utils_ import visual_position_map
            # visual_position_map(data['image_path'],pos_heatmap,'./tmp.jpg')
            # raise
            vessel_path=os.path.join(data_path,'vessel_seg',data['image_name'].split('.')[0]+'.png')
            data.update({
                'vessel_path':vessel_path,
                'pos_heatmap':position_save_path
            })
            annotate.append(data)
        with open(os.path.join(data_path,'ridge',f'{split}.json'),'w') as f:
            data_list=json.dump(annotate,f)


def parse_json(input_data,label_class=0,image_dict="../autodl-tmp/images"):
    annotations = input_data.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])
    image_name=input_data["file_upload"].split('-')[-1]
    new_data = {
        "image_name": image_name,
        "image_path":os.path.join(image_dict,image_name),
        "ridge_number": 0,
        "ridge_coordinate": [],
        "other_number": 0,
        "other_coordinate": [],
        "plus_number": 0,
        "plus_coordinate": [],
        "pre_plus_number": 0,
        "pre_plus_coordinate": [],
        "class": label_class
    }

    for item in result:
        if item["type"] == "keypointlabels":
            # x, y = item["value"]["x"], item["value"]["y"]
            x= item["value"]["x"]*item["original_width"]/100
            y= item["value"]["y"]*item["original_height"]/100
            label = item["value"]["keypointlabels"][0]

            if label == "Ji":
                new_data["ridge_number"] += 1
                new_data["ridge_coordinate"].append((x, y))
            elif label == "Other":
                new_data["other_number"] += 1
                new_data["other_coordinate"].append((x, y))
            elif label == "Plus":
                new_data["plus_number"] += 1
                new_data["plus_coordinate"].append((x, y))
            elif label == "Pre-plus":
                new_data["pre_plus_number"] += 1
                new_data["pre_plus_coordinate"].append((x, y))

    return new_data

def parse_json_file(file_dict,data_path):
    
    annotation=[]
    file_list=sorted(os.listdir(file_dict))
    print(f"read the origianl json file from {file_list}")
    for file in file_list:
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        with open(os.path.join(file_dict,file), 'r') as f:
            data = json.load(f)
        
        for json_obj in data:
            new_data=parse_json(json_obj,label_class=int(file[0]),
                                image_dict=os.path.join(data_path,'images'))
            if new_data["ridge_number"]>0:        
                annotation.append(new_data)

    return annotation

def split_data(data_path, annotations):
    # Important: do not shuffle, as there may be some images very similar be split into different set
    os.makedirs(os.path.join(data_path, 'ridge'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path, 'ridge')}/*")

    with open(os.path.join(data_path, 'annotations', "train.json"), 'r') as f:
        train_list=json.load(f)
        train_list=[i['image_name'] for i in train_list]
    with open(os.path.join(data_path, 'annotations', "val.json"), 'r') as f:
        val_list=json.load(f)
        val_list=[i['image_name'] for i in val_list]
    with open(os.path.join(data_path, 'annotations', "test.json"), 'r') as f:
        test_list=json.load(f)
        test_list=[i['image_name'] for i in test_list]

    train_annotations = []
    val_annotations = []
    test_annotations =[]
    train_condition={'1':0,"2":0,"3":0}
    val_condition={'1':0,"2":0,"3":0}
    test_condition={'1':0,"2":0,"3":0}
    for data in annotations:
        if data['image_name'] in train_list:
            train_annotations.append(data)
            train_condition[str(data['class'])]+=1
        if data['image_name'] in val_list:
            val_annotations.append(data)
            val_condition[str(data['class'])]+=1
        if data['image_name'] in test_list:
            test_annotations.append(data)
            test_condition[str(data['class'])]+=1
    with open(os.path.join(data_path, 'ridge', 'train.json'), 'w') as f:
        json.dump(train_annotations, f, indent=2)

    with open(os.path.join(data_path, 'ridge', 'val.json'), 'w') as f:
        json.dump(val_annotations, f, indent=2)

    with open(os.path.join(data_path, 'ridge', 'test.json'), 'w') as f:
        json.dump(test_annotations, f, indent=2)

    print(f"Total samples: {len(annotations)}"  )
    print(f"Train samples: {len(train_annotations)} {train_condition} {train_condition}")
    print(f"Validation samples: {len(val_annotations)} {val_condition} {val_condition}")
    print(f"Test samples: {len(test_annotations)} {test_condition} {test_condition}")

if __name__=='__main__':
    from config import get_config
    args=get_config()
    
    # cleansing
    if args.generate_ridge:
        annotations=parse_json_file(args.json_file_dict,args.path_tar)
        split_data(args.path_tar,annotations)
        print(f"generate ridge_coordinate in {os.path.join(args.path_tar,'ridge')}")
    if args.generate_diffusion_mask:
        generate_ridge_diffusion(args.path_tar)
    if args.generate_vessel:
        generate_vessel_result(data_path=args.path_tar)

    generate_possion_map(args.path_tar,args.posi_compress)