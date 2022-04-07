import torch
print(torch.cuda.is_available())

a = torch.zeros((3,3,10))
a[1,1,3] = 2.3
print(a[2,2,10]  )