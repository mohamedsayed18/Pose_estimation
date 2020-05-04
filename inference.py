from helpers import *

# Load
model = torch.load("./torch_model_v1.pt")
model.eval()

output = test(model, './Data_set/Prayer_pose/img0.jpg')
print(output)