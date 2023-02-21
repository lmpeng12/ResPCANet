# – coding: UTF-8
import math
import torch
import torch.nn as nn
from sklearn import svm
from torch.nn.parameter import Parameter
class PCANet(nn.Module):

    def __init__(self,k1 = 7,k2 = 7,L1 = 8,L2 = 8,input_channel = 10,
                 block_size = 8, overlapping_radio=0, linear_classifier='svm', dim_reduction=None):
        super().__init__()
        # Set the size k1 and k2 of the sliding window convolution kernel, and L1 and L2 are the size of the eigenvector obtained in each step
        # Number of convolution kernels in the first stage of L1
        # Number of convolution kernels in L2 second stage
        self.k1 = k1
        self.k2 = k2
        self.L1 = L1
        self.L2 = L2
        self.input_channel = input_channel
        self.block_size = block_size
        # Block size histogram
        self.overlapping_radio = overlapping_radio

        # Set the convolution of feature vectors extracted from l1 and l2
        self.l1_filters = None
        self.l2_filters = None
        #No offset required
        self.l1_conv = nn.Conv2d(in_channels=input_channel,
                                 out_channels=self.L1,
                                 kernel_size=[k1,k2],
                                 stride=1,padding=[k1 // 2, k2 //2],
                                 bias=False,
                                 )
        self.l2_conv = nn.Conv2d(in_channels=1,
                                 out_channels=self.L2,
                                 kernel_size=[k1,k2],
                                 stride=1,padding=[k1 // 2, k2 //2],
                                 bias=False)
        #The bootstrap convolution is not updated by calculation. Of course, this part can also be updated by this bootstrap operation
        self.l1_conv.requires_grad_(False)
        self.l2_conv.requires_grad_(False)
        #Whether to use linear classification SVM code
        if linear_classifier == 'svm':
            self.classifier = svm.SVC(probability=True)
        else:
            self.classifier = None

        # Reduce the dimension after spp to, optional, default to none
        if dim_reduction:
            self.dim_reduction = dim_reduction
        else:
            self.dim_reduction = None

        # ---------------- There are problems, so we will not consider using it for the time being——————————————
        # Here, referring to the operation similar to transformer, you can set the writing method of the convolution kernel
        # in which a single row and column exists, or you can learn directly
        # [1 0], [0,1,0], [0,0,1] and so on respectively retain the relevant position values
        # If the step is 1, you can read directly
        self.mean_conv1 = nn.Conv2d(in_channels=self.input_channel,
                                    out_channels=k1 * k2 * self.input_channel,
                                    kernel_size=(k1,k2),
                                    stride=1)
        # 赋值生成每个像素点的限制矩阵，并且不进行有关的学习过程
        if 1 == 1:
            mean_conv1_parameters = torch.eye(k1 * k2)
            mean_conv1_parameters = mean_conv1_parameters.reshape(1,k1*k2,k1,k2)
            # mean_conv1_parameters = mean_conv1_parameters.repeat(input_channel,1,1,1)
            #The assignment generates the limit matrix of each pixel point, and does not carry out the relevant learning process
            mean_conv1_parameters = mean_conv1_parameters.permute(1,0,2,3)
            mean_conv1_parameters = mean_conv1_parameters.repeat(input_channel,input_channel,1,1)
            for i in range(input_channel):
                mean_conv1_parameters_zeros = torch.zeros(size = mean_conv1_parameters.shape)
                mean_conv1_parameters_zeros[i * self.k1 * self.k2: (i+1) * self.k1 * self.k2,i:i+1,:,:] = 1
                mean_conv1_parameters = mean_conv1_parameters * mean_conv1_parameters_zeros



            self.mean_conv1.weight = Parameter(mean_conv1_parameters)

        self.non_Linear_fun = nn.ReLU(inplace=True)


    #Put the classifier in. Because the classification needs fast operation, we need to consider the mixed operation here.
    def setClassifierTool(self,classifier):
            self.classifier = classifier

    def forward(self,x,is_train = False):
        #These calculation processes can be improved
        if is_train:
            self.l1_filters = self.pca_induced_filter(x,out_put_channel=self.L1)
            self.l1_conv.weight = Parameter(self.l1_filters)
        out1 = self.l1_conv(x)

        num, ch, w, h = out1.shape
        out2 = out1.reshape(num*ch,1,w,h)
        if is_train:
            self.l2_filters = self.pca_induced_filter(out2,out_put_channel=self.L2)
            self.l2_conv.weight = Parameter(self.l2_filters)
        out2 = self.l2_conv(out2)
        out2 = out2.reshape(num,ch,self.L2,w,h)
        all_feature = self.extract_features_big_memory(out2)

        return all_feature


    #To remove the mean value, further processing can be considered here, including the processing can be optimized,
    # or the channel below can be expanded after
    def mean_remove_img_patches(self, img):
        # Block and subtract the mean
        # It is possible that only partial data is needed here
        out1 = self.mean_conv1(img) #number channel W H
        num,ch,w,h = out1.shape
        out1 = out1.reshape(num,ch,w*h)
        #Move the relevant channel to the rear
        out1 = out1.permute(0,2,1)

        out1 = out1.reshape(num, w * h, ch // self.input_channel, self.input_channel)
        cap_x_i = out1 - torch.mean(out1, dim=2, keepdim=True)
        cap_x_i = cap_x_i.permute(0, 3, 1, 2)  # num ch sqnum k1 * k2
        return cap_x_i


    #But the convolution kernel must be modified again
    def mean_remove_img_patches_base(self,img):
        # img : num ch w h
        #The original python writing method is improved.
        # After the block is directly divided, the dimension can be merged and expanded directly

        num, ch, w, h = img.shape
        width = img.shape[-2] - self.k1 + 1
        height = img.shape[-1] - self.k2 + 1
        out1 = []
        for i in range(width):
            for j in range(height):
                temp_img = (img[:,:,i: i + self.k1, j:j + self.k2]).reshape(num,ch,-1).unsqueeze(3)
                out1.append((img[:,:,i: i + self.k1, j:j + self.k2]).unsqueeze(4))

        #It doesn't need to expand multiple dimensions. Consider adding a dimension.
        # First, observe the writing method. If the dimension part, you can operate it together

        out1 = torch.cat(out1,dim=4)
        out1 = out1.reshape(num,ch,self.k1 * self.k2,width * height)
        cap_x_i = out1 - torch.mean(out1, dim=2, keepdim=True)
        return cap_x_i


    # 这一块后期再写成一个block进行处理，并且独立成为一个类别
    def pca_induced_filter(self,train_data,out_put_channel):
        '''
        :param train_data:
        :param out_put_channel:
        :return:
        # 分块的大小
        patch_width = self.k1 # 这个每次的选取卷积核大小都不固定，因此这里边还是需要说稍微考究一下的
        patch_height = self.k2
        # 计算分块的列长度处理，这里默认了行移动和列移动都是1
        img_patch_width = w - patch_width + 1
        img_patch_height = h - patch_height + 1
        '''
        num,ch,w,h = train_data.shape

        # Just choose one from two
        # cap_x = self.mean_remove_img_patches(train_data)
        cap_x = self.mean_remove_img_patches_base(train_data)

        # cap_c = torch.matmul(cap_x,cap_x.permute(0,1,3,2))
        _,_,lin_kernel,split_num = cap_x.shape
        cap_x = cap_x.reshape(num,ch * lin_kernel,split_num)
        cap_c = torch.matmul(cap_x,cap_x.permute(0,2,1))

        # sum
        cap_c = torch.sum(cap_c,dim = 0)

        # Calculate eigenvalues and eigenvectors
        # Torch.linalg. has tools such as svd in its library to calculate eigenvalues and vectors
        vals,vecs = torch.eig(cap_c / (num * split_num),eigenvectors=True)
        # Take the first k principal components as the convolution kernel.
        # Since there may be complex numbers, that is, a+bi, we consider directly selecting the values of the real number field
        # idx_ w = torch.argsort(torch.real(vals))[:-(out_put_channel + 1):-1]
        # Take out the first k main eigenvalues
        idx_w = torch.argsort(vals[:,0],descending=True)[:out_put_channel]

        cap_w = vecs[:,idx_w]
        # filters_ Data shape: [out_put, in_put, k1, k2] You need to transpose here first
        filters_data = cap_w.T.reshape(out_put_channel,ch,self.k1,self.k2)
        return filters_data

    #Directly operate the whole matrix without memory saving operation
    def extract_features_big_memory(self,x):
        #The output of binarization is assumed to be num L1 L2 w h
        num,L1,L2,w,h = x.shape
        binary_result = self.heaviside(x)
        decimal_result = torch.zeros(num,L1,w,h).to(device = x.device)
        for i in range(self.L2):
            decimal_result += (2 ** i) * binary_result[:, :, i, :, :]

        histo_bins = 2 ** self.L2
        img_width, img_height = decimal_result.shape[-2], decimal_result.shape[-1]
        #This is only about block
        step_size = int(self.block_size * (1 - self.overlapping_radio))
        img_patch_width = img_width - self.block_size + 1
        img_patch_height = img_height - self.block_size + 1

        #You can use other statistical planning intervals to process
        all_data_feature = []
        patten_split = []
        for i in range(0, img_patch_width, step_size):
            for j in range(0, img_patch_height, step_size):
                patten_split.append((decimal_result[:, :, i: i + self.block_size, j:j + self.block_size]).unsqueeze(2))
        patten_split = torch.cat(patten_split, dim=2)
        patten_split = patten_split.reshape(num, L1, patten_split.shape[2],
                                            patten_split.shape[-2] * patten_split.shape[-1])

        all_data_feature = []
        divide = 4  #Here, the data is sorted with 4 steps
        for i in range(0, histo_bins, divide):
            temp_patten_split = torch.sum((patten_split == i).float(), dim=3, keepdim=True)
            for j in range(1, divide):
                temp_patten_split += torch.sum((patten_split == (i + j)).float(), dim=3, keepdim=True)
            all_data_feature.append(temp_patten_split)
        all_data_feature = torch.cat(all_data_feature, dim=3)

        #Expand all to one dimension and keep the results
        return all_data_feature.reshape(num, -1)


    def heaviside(self,x):
        return (x>0).int()

    def normal_round(self,n):
        if n - torch.floor(n) < 0.5:
            return torch.floor(n)
        return math.ceil(n)


if __name__ == '__main__':

    if 1 == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        testdata2 = torch.rand(1000, 8,8, 25, 25) - 0.5
        net = PCANet(input_channel=3)
        net = net.to(device)
        testdata2 = testdata2.to(device)
        feature = net(testdata2, is_train=True)
        print(feature)
