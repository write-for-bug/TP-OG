from data import OODDataset,ClassOrderSampler
from torch.utils.data import DataLoader
if __name__ == '__main__':
  dataset = OODDataset(
    root='./datasets/ImageNet100',
    split='train',
    subset=None,
    return_type='path'
  )
  sampler = ClassOrderSampler(dataset, batch_size=4)
  dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, shuffle=False)
  old_label = [0]
  k = 0
  for path,label in dataloader:
    if label[0]!=old_label[0]:
      k+=1
      print(k,path,label[0])
    old_label = label