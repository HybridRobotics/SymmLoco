from escnn import gspaces                                          #  1
from escnn import nn                                               #  2
import torch                                                       #  3
                                                                   #  4
s = gspaces.flip2dOnR2() 
cnn_feat_type_in = nn.FieldType(s, [s.trivial_repr])
cnn_feat_type_hidden = nn.FieldType(s, 3*[s.regular_repr])
cnn_feat_type_out = nn.FieldType(s, 5*[s.regular_repr])
ecnn1 = nn.R2Conv(cnn_feat_type_in, cnn_feat_type_hidden, kernel_size=7)
ReLU1 = nn.ReLU(cnn_feat_type_hidden)
MaxPool = nn.PointwiseMaxPool(cnn_feat_type_hidden, kernel_size=2, stride=2)
ecnn2 = nn.R2Conv(cnn_feat_type_hidden, cnn_feat_type_out, kernel_size=6)
ReLU2 = nn.ReLU(cnn_feat_type_out)
GroupPool = nn.GroupPooling(cnn_feat_type_out)

x = torch.randn(2, 1, 32, 32)
x = cnn_feat_type_in(x)
cnn_hidden = ecnn1(x)
cnn_hidden = ReLU1(cnn_hidden)
cnn_hidden = MaxPool(cnn_hidden)

cnn_out = ecnn2(cnn_hidden)
cnn_out = ReLU2(cnn_out)
# cnn_out = cnn_out.tensor
# cnn_out = torch.mean(cnn_out, dim=(2,3))
cnn_out = GroupPool(cnn_out)

flip_x = torch.flip(x.tensor, dims=[3])
flip_x = cnn_feat_type_in(flip_x)
flip_cnn_hidden = ecnn1(flip_x)
flip_cnn_hidden = ReLU1(flip_cnn_hidden)
flip_cnn_hidden = MaxPool(flip_cnn_hidden)

flip_cnn_out = ecnn2(flip_cnn_hidden)
flip_cnn_out = ReLU2(flip_cnn_out)
# flip_cnn_out = flip_cnn_out.tensor
# flip_cnn_out = torch.mean(flip_cnn_out, dim=(2,3))
flip_cnn_out = GroupPool(flip_cnn_out)

print(cnn_out)
print(flip_cnn_out)