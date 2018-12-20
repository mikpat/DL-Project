from model.u_net import get_unet_1024_src_3, get_unet_1024_src_5, get_unet_1024_gcn_down_k_5, get_unet_1024_gcn_up_k_5,  get_unet_1024_gcn_down_up_k_5

input_size = 1024

max_epochs = 100
batch_size = 16

orig_width = 1918
orig_height = 1280

threshold = 0.5


model_factory = [get_unet_1024_src_3()]

model_names = ["unet_1024_src_3"]

'''
model_factory = [get_unet_1024_src_3(), get_unet_1024_src_5(), get_unet_1024_gcn_down_k_5(),
                 get_unet_1024_gcn_up_k_5(), get_unet_1024_gcn_down_up_k_5()]

model_names = ["unet_1024_src_3", "unet_1024_src_5", "unet_1024_gcn_down_k_5",
                "unet_1024_gcn_up_k_5", "unet_1024_gcn_down_up_k_5"]
'''
