import torch
import torchvision.models as models

def load_simsiam(model_name):
    if model_name == 'simsiam-rn50':
        path = 'X:/Trained_model_storage/SSL Images/checkpoints/rn50/simsiam.pth.tar'
        model = models.__dict__['resnet50']()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()


    # load from pre-trained, before DistributedDataParallel constructor

    checkpoint = torch.load(path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    return model
