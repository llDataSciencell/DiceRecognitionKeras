import os
from PIL import Image
import glob


UNZIP_DATA_PATH=os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset"

def make_dirs(labels):
  for kind in ('train', 'valid', 'test'):
    for label in labels:
      os.makedirs('%s/%s/%s' % (UNZIP_DATA_PATH, kind, label),
                  mode=0o775, exist_ok=True)

def convert_and_save(infile, outfile):
  image = Image.open(infile)
  image = image.resize((128, 128))
  image.save(outfile)

def write_image(labels):
  '''Splitting and rescaling images into train and valid datasets.'''
  train_valid_test_ratio=[0.7,0.15,0.15]

  valid_test_flag = ["train", "valid", "test"]
  for label in labels:
    categorical_total_file=len(glob.glob('%s/%s/*.bmp' % (UNZIP_DATA_PATH,label)))
    idx=0
    valid_test_flag_idx = 0
    for infile in sorted(glob.glob('%s/%s/*.bmp' % (UNZIP_DATA_PATH,label))):
      idx+=1
      if idx > categorical_total_file * train_valid_test_ratio[valid_test_flag_idx]:
          valid_test_flag_idx+=1
          idx=0
      outfile = '/%s/%s/%s/%s' % (UNZIP_DATA_PATH,valid_test_flag[valid_test_flag_idx],label,os.path.basename(infile)[:-3]+"png")
      print("in:" + str(infile))
      print("out:"+str(outfile))
      convert_and_save(infile, outfile)

labels=["1","2","3","4","5","6"]
make_dirs(labels)
write_image(labels)