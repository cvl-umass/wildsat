from models.satclip.main import *

def get_satclip(ckpt_path, return_all=True):
    # ckpt = torch.load(ckpt_path,map_location=device)
    ckpt = torch.load(ckpt_path, map_location="cuda")
    ckpt['hyper_parameters'].pop('eval_downstream')
    ckpt['hyper_parameters'].pop('air_temp_data_path')
    ckpt['hyper_parameters'].pop('election_data_path')
    lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to('cuda')

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()
    
    geo_model = lightning_model.model
    return geo_model
    # if return_all:
    #     return geo_model
    # else:
    #     return geo_model.location