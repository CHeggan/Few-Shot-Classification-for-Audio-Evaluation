
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

from dataset_.FeatureExtractor import FeatureExtractor
from FewShotClassification import FewShotClassification
from models_.encoder_selection import resnet_selection, adapter_resnet_selection, split_resnet_selection

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
    
    # Selects the classes of interest (coi) based on split specified
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
    if 'adapter' in params['model']['name']:
        fc_list = [params['model'][ params['model']['name'] ]['head_dim']] * \
            params['model'][ params['model']['name'] ]['num_heads']

        model = adapter_resnet_selection(dims=params['data']['in_dims'],
            fc_out_list=fc_list,
            in_channels=params['data']['in_channels'],
            task_mode=params['model'][ params['model']['name'] ]['task_mode'],
            num_tasks=params['model'][ params['model']['name'] ]['num_heads'],
            model_name=params['model']['name'])

        extra_params = {'task_int': 'all'}

    elif 'split' in params['model']['name']:
        fc_list = [params['model'][ params['model']['name'] ]['head_dim']] * \
            params['model'][ params['model']['name'] ]['num_heads']

        model = split_resnet_selection(dims=params['data']['in_dims'],
            fc_out=fc_list,
            in_channels=params['data']['in_channels'],
            model_name=params['model']['name'])

        extra_params = {'task_int': 'all'}

    else:
        model = resnet_selection(dims=params['data']['in_dims'], 
            model_name=params['model']['name'], 
            fc_out=params['model'][ params['model']['name'] ]['encoder_fc_dim'], 
            in_channels=params['data']['in_channels'])
        extra_params = {}

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

    if params['use_feats'][data_name] == True:

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

        features, labels = feat_generator.generate(params['ft_params'], verbose=True, 
            extra_params=extra_params)


    #########################
    # FEW-SHOT PROBLEM
    #########################
    if params['use_feats'][data_name] == True:
        fs_dataset = FullSetWrapper(full_set=features, labels=labels.cpu())
    # If we dont use the pre-compute features, we re-use the previous path finding dataset
    else:
        fs_dataset = dataset


    results = []
    for hardness in params['task']['hardness']:

        if ('adapter' in params['model']['name']) or ('split' in params['model']['name']):
            model_fc_out=params['model'][ params['model']['name'] ]['head_dim'] * params['model'][ params['model']['name'] ]['num_heads']
        else:
            model_fc_out = params['model'][ params['model']['name'] ]['encoder_fc_dim']

        fs_class = FewShotClassification(dataset=fs_dataset,
            params=params,
            model_fc_out=model_fc_out,
            device=device,
            variable=data_params['target_data']['variable'],
            hardness=hardness,
            additional_fn=extra_batch_work
            )


        if params['use_feats'][data_name] == True:
            accs = fs_class.eval_w_feats()

        else:
            accs = fs_class.eval_w_samples(model=model,
                additional_batch_fn=extra_batch_work,
                extra_model_params=extra_params)


        mean = np.mean(accs)
        std = np.std(accs)
        ci_95 = (1.96 / np.sqrt(int(params['task']['num_tasks']))) * std *100


        result_dict = {'model_name': model_file_path.split('.')[0].split('/')[-1],
            'dataset': data_params['target_data']['name'], 
            'hardness': hardness,
            'mean': mean,
            'std': std,
            'CI_95': ci_95}

        results.append(result_dict)

        print('\n',data_params['target_data']['name'], mean, '\n')
    return results
