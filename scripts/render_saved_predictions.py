#!/usr/bin/env python3
"""Render saved COCO detections over fixed image samples for presentation."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

DOMAINS=("day_clear","day_foggy","dusk_rainy","night_clear","night_rainy")
COLORS={"bus":(65,145,230),"bike":(214,90,196),"car":(28,180,105),"motor":(237,149,43),"person":(230,81,81),"rider":(50,190,195),"truck":(147,104,220)}
W,H,HEADER=640,400,48

def font(size):
 p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
 return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--annotations-root',type=Path,required=True)
 ap.add_argument('--baseline-dir',type=Path,required=True)
 ap.add_argument('--moe-dir',type=Path,required=True)
 ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--min-instances',type=int,default=3)
 ap.add_argument('--score-threshold',type=float,default=.05)
 ap.add_argument('--max-boxes',type=int,default=20)
 a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
 summary={'source':'saved job 15681 test predictions; rendered after frozen evaluation','split':'test','selection_rule':f'lowest image_id with >= {a.min_instances} non-crowd GT boxes','score_threshold':a.score_threshold,'domains':{}}
 rows={}
 for domain in DOMAINS:
  coco=json.loads((a.annotations_root/domain/f'{domain}_test.coco.json').read_text())
  categories={c['id']:c['name'] for c in coco['categories']}; images={i['id']:i for i in coco['images']}; gt=defaultdict(list)
  for ann in coco['annotations']:
   if not ann.get('iscrowd',0):gt[ann['image_id']].append(ann)
  ids=[i for i in sorted(images) if len(gt[i])>=a.min_instances]
  if not ids:raise RuntimeError(f'No suitable test image: {domain}')
  image_id=ids[0]; info=images[image_id]; imgpath=Path(info['file_name'])
  if not imgpath.is_file():raise FileNotFoundError(imgpath)
  base=json.loads((a.baseline_dir/f'baseline_predictions_{domain}.json').read_text())
  moe=json.loads((a.moe_dir/f'moe_predictions_{domain}.json').read_text())
  base=[x for x in base if x['image_id']==image_id and x['score']>=a.score_threshold]
  moe=[x for x in moe if x['image_id']==image_id and x['score']>=a.score_threshold]
  bgr=cv2.imread(str(imgpath)); image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
  panels=[]
  for title,boxes,isgt in [('Ground truth',gt[image_id],True),('Baseline · 1k cycle',base,False),('MoE · 1k cycle',moe,False)]:
   panel=Image.new('RGB',(W,H+HEADER),'white'); d=ImageDraw.Draw(panel); d.rectangle((0,0,W,HEADER),fill=(23,37,31)); d.text((14,13),title,font=font(21),fill='white')
   scale=min(W/image.width,H/image.height); small=image.resize((int(image.width*scale),int(image.height*scale)))
   xo=(W-small.width)//2; yo=HEADER+(H-small.height)//2; panel.paste(small,(xo,yo)); d=ImageDraw.Draw(panel)
   for item in sorted(boxes,key=lambda x: x.get('score',1),reverse=True)[:a.max_boxes]:
    cid=item['category_id']; name=categories[cid]; color=COLORS.get(name,(255,220,50))
    if isgt:
     x,y,bw,bh=item['bbox']; x2=x+bw; y2=y+bh; label=name
    else:
     x,y,bw,bh=item['bbox']; x2=x+bw; y2=y+bh; label=f"{name} {item['score']:.2f}"
    coords=(xo+x*scale,yo+y*scale,xo+x2*scale,yo+y2*scale)
    d.rectangle(coords,outline=color,width=3); ty=max(yo,coords[1]-20); tb=d.textbbox((coords[0],ty),label,font=font(15)); d.rectangle(tb,fill=color); d.text((coords[0],ty),label,font=font(15),fill=(10,20,16))
   panels.append(panel)
  row=Image.new('RGB',(W*3,H+HEADER),'white')
  for i,panel in enumerate(panels):row.paste(panel,(i*W,0))
  row.save(a.output_dir/f'{domain}.png',optimize=True); rows[domain]=row
  summary['domains'][domain]={'image_id':image_id,'image_file':imgpath.name,'gt_count':len(gt[image_id]),'baseline_predictions_above_threshold':len(base),'moe_predictions_above_threshold':len(moe)}
  print(f'rendered {domain} image_id={image_id} GT={len(gt[image_id])} baseline={len(base)} MoE={len(moe)}')
 sheet=Image.new('RGB',(W*3,(H+HEADER)*len(DOMAINS)),'white')
 for i,domain in enumerate(DOMAINS):sheet.paste(rows[domain],(0,i*(H+HEADER)))
 sheet.save(a.output_dir/'five_domain_contact_sheet.jpg',quality=88,optimize=True)
 highlight=Image.new('RGB',(W*3,(H+HEADER)*2),'white')
 for i,domain in enumerate(('day_foggy','night_clear')):highlight.paste(rows[domain],(0,i*(H+HEADER)))
 highlight.save(a.output_dir/'presentation_dayfoggy_nightclear.jpg',quality=90,optimize=True)
 (a.output_dir/'samples.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
