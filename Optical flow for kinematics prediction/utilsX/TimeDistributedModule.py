class TimeDistributed(nn.Module):
    def __init__(self, module, time_steps, isCuda):        
        super(TimeDistributed, self).__init__()
        
        self.module = nn.ModuleList([module for i in range(time_steps)])

    def forward(self, x):
        #print(x.is_cuda)
        time_steps, batch_size, C, H, W = x.size()
        #print("INPUT Image Tensor Size = ",x.size())
        #print(next(self.module.parameters()).is_cuda)
        output = torch.tensor([])
        if isCuda:
            output = output.to('cuda')
        for i in range(time_steps):
            #print("time step = ", i+1)
            #print("------------------------------------------------------")
            output_t = self.module[i](x[i, :, :, :, :])
            #print("Resnet feature map size = ",output_t.size())
            output_t  = output_t.unsqueeze(0)
            #print("Unsqueezed Resnet feature map size = ",output_t.size())
            output = torch.cat((output, output_t ), 0)
            #print("timedistributed output size = ",output.size())
            #print("------------------------------------------------------")
        return output

