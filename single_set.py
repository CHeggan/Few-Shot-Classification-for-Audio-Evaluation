
###############################################################################
# IMPORTS
###############################################################################
import numpy as np

from utils import load_backbone
from dataset_.prep_batch_fns import basic_flexible_prep_batch
from dataset_.dataset_classes.FastDataLoader import FastDataLoader
from dataset_.dataset_classes.PathFinderSet import PathFinderSet
from dataset_.dataset_classes.FullSetWrapper import FullSetWrapper

from dataset_.dataset_utils import variable_collate, batch_to_log_mel_spec, \
    batch_to_log_mel_spec_plus_stft, nothing_func, variable_collate

from models_.encoder_selection import resnet_selection
from dataset_.FeatureExtractor import FeatureExtractor
from FewShotClassification import FewShotClassification

###############################################################################
# SINGLE DATASET RUN
###############################################################################
def single_dataset_run(params, data_params, model_file_path, device):

    #########################
    # INITIAL DATASET HANDLING 
    #########################
    data_name = data_params['target_data']['name']
    train_classes, val_classes, test_classes = np.load(data_params['target_data']['fixed_path'],
        allow_pickle=True)
    
    # Selects teh classes of interest (coi) based on split specified
    if params['task']['split'] == 'val':
        coi = val_classes
    elif params['task']['split'] == 'train':
        coi = train_classes
    elif params['task']['split'] == 'test':
        coi = test_classes

    dataset = PathFinderSet(classes=coi,
        class_root_dir=params['data_paths'][data_name],
        ext= params['data']['ext'])

    #########################
    # MODEL SELECTION
    #########################
    model = resnet_selection(dims=params['data']['in_dims'], 
        model_name=params['model']['name'], 
        fc_out=params['model'][ params['model']['name'] ]['encoder_fc_dim'], 
        in_channels=params['data']['in_channels'])

    model = load_backbone(model, model_file_path, verbose=True)
    model = model.to(device)
    model.eval()

    #########################
    # ADDITIONAL DATA FUNCS
    #########################
    if params['data']['in_dims'] == 1 and params['data']['in_channels'] ==1:
        extra_batch_work = nothing_func
    elif params['data']['in_dims'] == 2 and params['data']['in_channels'] ==1:
        extra_batch_work = batch_to_log_mel_spec
    elif params['data']['in_dims'] == 2 and params['data']['in_channels'] ==3:
        extra_batch_work = batch_to_log_mel_spec_plus_stft
    else:
        raise ValueError('Thaaaaaaanks, an incorrect configuration file')

    #########################
    # COLLATION FUNCTION
    #########################
    # for variable length we need a different collaction function
    # Defines the datasets to be used
    if data_params['target_data']['variable']:
        col_fn = variable_collate
    else:
        col_fn = None

    #########################
    # FEATURE EXTRACTION
    #########################
    # We generate features on a batch_size=1 basis and then store them in a list 
    #   of tensors. Doing so is mainly for variable sets. It prevents the need for
    #   either the re-splitting of stacked tensors or averaging across multi-sequence 
    #   samples (which would result in info loss)

    flat_batcher = basic_flexible_prep_batch(device=device, data_type='float',
        variable=data_params['target_data']['variable'])

    flat_dataloader = FastDataLoader(dataset, 
        batch_size=params['extraction']['batch_size'],
        num_workers=params['extraction']['num_workers'],
        shuffle=False,
        collate_fn=col_fn)

    feat_generator = FeatureExtractor(dataloader=flat_dataloader,
        model=model,
        prep_batch=flat_batcher,
        additional_fn=extra_batch_work,
        variable=data_params['target_data']['variable'])

    features, labels = feat_generator.generate(params['ft_params'], verbose=True)

    print(len(features), labels.shape)

    #########################
    # FEW-SHOT PROBLEM
    #########################
    fs_dataset = FullSetWrapper(full_set=features, labels=labels.cpu())



    for hardness in params['task']['hardness']:

        print(hardness)

        fs_class = FewShotClassification(dataset=fs_dataset,
            params=params,
            model_fc_out=params['model'][ params['model']['name'] ]['encoder_fc_dim'],
            device=device,
            variable=data_params['target_data']['variable'],
            hardness=hardness,
            additional_fn=extra_batch_work
            )

        accs = fs_class.eval()
