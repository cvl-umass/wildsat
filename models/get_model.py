import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, resnet101, swin_t, vit_b_16
import satlaspretrain_models	# pip install
import torch
from torch import nn

from loguru import logger
import sinr.models
import sinr.datasets
import sinr.utils
from models.seco import MocoV2

def get_model(
	encoder_name, head, num_outputs, task="classification", num_inp_feats=3, 
	pretrained=True, encoder_ckpt_path=None, is_residual=False, normalize=False,
	use_lora=False, lora_layer_types=["conv"], use_dora=False,
):
	logger.debug(f"pretrained: {pretrained}")
	w = "IMAGENET1K_V2" if pretrained else None
	logger.debug(f"using weight w: {w}")
	# Return multi-task learning model or single-task model
	is_satlas = False
	encoder_name_for_fwd = None
	if encoder_name == "resnet50":
		encoder = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
		num_feats = encoder.fc.out_features
		if pretrained and (encoder_ckpt_path is not None):
			if "satclip" in encoder_ckpt_path.lower():
				from models.satclip.main import SatCLIPLightningModule
				satclip_params = torch.load(encoder_ckpt_path, map_location='cpu')
				satclip_params['hyper_parameters'].pop('eval_downstream')
				satclip_params['hyper_parameters'].pop('air_temp_data_path')
				satclip_params['hyper_parameters'].pop('election_data_path')
				lightning_model = SatCLIPLightningModule(**satclip_params['hyper_parameters'])
				encoder = lightning_model.model.visual
				num_feats = encoder.fc.out_features
				logger.debug(f"num_feats: {num_feats}. Using satclip encoder.")
			elif "satlas-backbone" in encoder_ckpt_path.lower():
				encoder = SatlasModelBackbone(num_inp_feats=num_inp_feats, model_name="Sentinel2_Resnet50_SI_RGB")
				num_feats = encoder.backbone_channels
				logger.debug(f"num_feats: {num_feats}. Using satlas-backbone. is_satlas={is_satlas}")
			elif "seco" in encoder_ckpt_path.lower():
				seco_model = MocoV2.load_from_checkpoint(encoder_ckpt_path)
				encoder = seco_model.encoder_q
				num_feats = encoder[-3][-1].bn3.num_features
				logger.debug(f"num_feats: {num_feats}. Using seco. encoder_ckpt_path={encoder_ckpt_path}")
			elif "moco" in encoder_ckpt_path.lower():
				state_dict = torch.load(encoder_ckpt_path)['state_dict']
				linear_keyword = 'fc'
				for k in list(state_dict.keys()):	# from MoCo (https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_lincls.py#L179)
					# retain only base_encoder up to before the embedding layer
					if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
						# remove prefix
						state_dict[k[len("module.base_encoder."):]] = state_dict[k]
				msg = encoder.load_state_dict(state_dict, strict=False)
				logger.debug(f"missing keys: {msg.missing_keys} [from loading {encoder_ckpt_path}]")
				num_feats = encoder.fc.out_features
			elif "imagenetv1" in encoder_ckpt_path.lower():
				logger.warning(f"Using old imagenet pretrained weights IMAGENET1K_V1")
				encoder = resnet50(weights="IMAGENET1K_V1")
				num_feats = encoder.fc.out_features
			elif "remoteclip" in encoder_ckpt_path.lower():
				import open_clip
				model_name = 'RN50' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
				model, _, preprocess = open_clip.create_model_and_transforms(model_name)
				ckpt = torch.load(f"checkpoints/remoteclip/RemoteCLIP-{model_name}.pt", map_location="cpu")
				message = model.load_state_dict(ckpt)
				logger.debug(f"Using pre-trained RemoteCLIP model: {message}")
				encoder = model.visual
				num_feats = 1024
		last_layer_names = ["fc"]	# for lora (will still train, but not use Lora)
	elif encoder_name == "resnet101":
		encoder = resnet101(weights="IMAGENET1K_V2" if pretrained else None)
		num_feats = encoder.fc.out_features
	elif encoder_name == "vitb16":
		encoder = vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
		num_feats = encoder.heads.head.out_features
		if pretrained and (encoder_ckpt_path is not None):
			if "graft" in encoder_ckpt_path.lower():
				ckpt_path = "checkpoints/graft/sentinel_model.ckpt"
				sd = torch.load(ckpt_path)
				encoder = GRAFT(temp=True, bias_projector=False)
				tmp = encoder.load_state_dict(sd['state_dict'], strict=False)
				logger.debug(f"model checkpoint loading: {tmp} [from {ckpt_path}]")
				# logger.debug(f"encoder: {encoder}")
				num_feats = 512
				encoder_name_for_fwd = "graft"
				logger.debug(f"num_feats: {num_feats}. Using GRAFT encoder. encoder_name_for_fwd={encoder_name_for_fwd}")
			elif "satclip" in encoder_ckpt_path.lower():
				from models.satclip.main import SatCLIPLightningModule
				satclip_params = torch.load(encoder_ckpt_path, map_location='cpu')
				satclip_params['hyper_parameters'].pop('eval_downstream')
				satclip_params['hyper_parameters'].pop('air_temp_data_path')
				satclip_params['hyper_parameters'].pop('election_data_path')
				lightning_model = SatCLIPLightningModule(**satclip_params['hyper_parameters'])
				encoder = lightning_model.model.visual
				num_feats = encoder.head.out_features
				logger.debug(f"num_feats: {num_feats}. Using satclip encoder.")
			elif "clip" in encoder_ckpt_path.lower():
				from transformers import CLIPVisionModelWithProjection
				CLIP_version="openai/clip-vit-base-patch16"
				encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
				num_feats = 512
				encoder_name_for_fwd = "clip"
				logger.debug(f"num_feats: {num_feats}. Using CLIP encoder. encoder_name_for_fwd={encoder_name_for_fwd}")
			elif "taxabind" in encoder_ckpt_path.lower():
				from transformers import PretrainedConfig
				from rshf.taxabind import TaxaBind
				logger.debug(f"Using pre-trained taxabind model")
				config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
				taxabind = TaxaBind(config)
				encoder = taxabind.get_sat_encoder()
				num_feats = 512
				encoder_name_for_fwd = "taxabind"
				logger.debug(f"num_feats: {num_feats} [from MVRL/taxabind]. encoder_name_for_fwd: {encoder_name_for_fwd}")
			elif "moco" in encoder_ckpt_path.lower():
				from models.moco_vit import vit_base
				encoder = vit_base()
				state_dict = torch.load(encoder_ckpt_path)['state_dict']
				linear_keyword = 'head'
				for k in list(state_dict.keys()):	# from MoCo (https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_lincls.py#L179)
					# retain only base_encoder up to before the embedding layer
					if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
						# remove prefix
						state_dict[k[len("module.base_encoder."):]] = state_dict[k]
				msg = encoder.load_state_dict(state_dict, strict=False)
				logger.debug(f"missing keys: {msg.missing_keys} [from loading {encoder_ckpt_path}]")
				num_feats = encoder.head.out_features
			elif "prithvi" in encoder_ckpt_path.lower():
				from models.prithvi import MaskedAutoencoderViT
				import yaml

				weights_path = "checkpoints/prithvi/Prithvi_100M.pt"
				model_cfg_path = "checkpoints/prithvi/Prithvi_100M_config.yaml"
				prithvi_checkpoint = torch.load(weights_path, map_location="cpu")
				with open(model_cfg_path) as f:
					model_config = yaml.safe_load(f)
				model_args, train_args = model_config["model_args"], model_config["train_params"]
				model_args["num_frames"] = 1
				encoder = MaskedAutoencoderViT(**model_args)
				del prithvi_checkpoint['pos_embed']
				del prithvi_checkpoint['decoder_pos_embed']
				tmp = encoder.load_state_dict(prithvi_checkpoint, strict=False)
				logger.debug(f"Using prithvi-100M encoder. missing keys: {tmp}")
				encoder.decoder_norm = nn.Identity()
				encoder.decoder_pred = nn.Identity()
				encoder.decoder_blocks = nn.Identity()
				encoder.decoder_embed = nn.Identity()
				logger.debug(f"Removed decoder blocks in Prithvi since not needed (changed to identity)")
				num_feats = 150528	# (196,14,14) flattened

		last_layer_names = ["head"]	# for lora (will still train, but not use Lora)
	elif encoder_name == "swint":
		encoder = swin_t(weights="IMAGENET1K_V1" if pretrained else None)
		num_feats = encoder.head.out_features
		if pretrained and (encoder_ckpt_path is not None):
			if "satlas-backbone" in encoder_ckpt_path.lower():
				encoder = SatlasModelBackbone(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinT_SI_RGB")
				num_feats = encoder.backbone_channels
				logger.debug(f"num_feats: {num_feats}. Using satlas-backbone. is_satlas={is_satlas}")
		last_layer_names = ["head"]	# for lora (will still train, but not use Lora)
	elif encoder_name == "swinb":
		if pretrained and (encoder_ckpt_path is not None):
			if "satlas-backbone" in encoder_ckpt_path.lower():
				encoder = SatlasModelBackbone(num_inp_feats=num_inp_feats, model_name="Sentinel2_SwinB_SI_RGB")
				num_feats = encoder.backbone_channels
				logger.debug(f"num_feats: {num_feats}. Using satlas-backbone. is_satlas={is_satlas}")
				# logger.debug(f"encoder: {encoder}")
	elif encoder_name == "vitb32":
		if pretrained and (encoder_ckpt_path is not None):
			if "remoteclip" in encoder_ckpt_path.lower():
				import open_clip
				model_name = 'ViT-B-32' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
				model, _, preprocess = open_clip.create_model_and_transforms(model_name)
				ckpt = torch.load(f"checkpoints/remoteclip/RemoteCLIP-{model_name}.pt", map_location="cpu")
				message = model.load_state_dict(ckpt)
				logger.debug(f"Using pre-trained RemoteCLIP model: {message}")
				encoder = model.visual
				num_feats = 512
	elif encoder_name == "vitl16":
		if pretrained and (encoder_ckpt_path is not None):
			if "satmae" in encoder_ckpt_path.lower():
				from models.satmae_vit import ViTFinetune, load_from_checkpoint
				target_size = 670	# for SatBird
				encoder = ViTFinetune(img_size=224, patch_size=16, in_chans=3, num_classes=target_size, embed_dim=1024,
										depth=24, num_heads=16, mlp_ratio=4, drop_rate=0.1, )
				encoder = load_from_checkpoint("checkpoints/satmae/fmow_pretrain.pth", encoder)
				num_feats = encoder.fc.in_features
				logger.debug(f"num_feats: {num_feats}. Using satmae. is_satlas={is_satlas} [checkpoints/satmae/fmow_pretrain.pth]")
				encoder.fc = nn.Identity()
				logger.debug(f"Changed last linear layer to identity since not necessary")
	elif encoder_name == "resnet18":
		if pretrained and (encoder_ckpt_path is not None):
			if "seco" in encoder_ckpt_path.lower():
				seco_model = MocoV2.load_from_checkpoint(encoder_ckpt_path)
				encoder = seco_model.encoder_q
				num_feats = encoder[-3][-1].bn1.num_features
				logger.debug(f"num_feats: {num_feats}. Using seco RESNET18. encoder_ckpt_path={encoder_ckpt_path}")

	else:
		raise NotImplementedError(f"model {encoder_name} not implemented")

	if use_lora or use_dora:
		from peft import LoraConfig, get_peft_model
		logger.debug(f"Using lora. Modifying encoder. lora_layer_types={lora_layer_types}. use_dora={use_dora}")
		lora_param_names = []
		for n, m in encoder.named_modules():
			for name in lora_layer_types:
				if name in n:
					lora_param_names.append(n)
		logger.debug(f"Found {len(lora_param_names)} layers to use lora on. modules_to_save={last_layer_names}")
		config = LoraConfig(
			target_modules=lora_param_names,
			modules_to_save=last_layer_names,
			use_dora=use_dora,
		)
		peft_model = get_peft_model(encoder, config)
		peft_model.print_trainable_parameters()
		encoder = peft_model


	logger.debug(f"num_feats: {num_feats}")
	if task == 'classification':
		decoder = get_classification_head(head, num_feats, num_outputs, is_residual=is_residual, normalize=normalize,is_satlas=is_satlas)
		model = SingleTaskModel(encoder, decoder, num_feats, encoder_name=encoder_name_for_fwd, num_inp_feats=num_inp_feats)

	return model


def get_classification_head(head, num_feats, num_outputs, is_residual=False, normalize=False,is_satlas=False):
	""" Return the decoder head """
	if head == 'mlp':
		return MLPModel(num_feats, num_outputs, normalize=normalize, is_satlas=is_satlas)
	elif head == "linear":
		return LinearModel(num_feats, num_outputs, bias=True, normalize=normalize, is_satlas=is_satlas)
	elif head == "linearnobias":
		return LinearModel(num_feats, num_outputs, bias=False, normalize=normalize, is_satlas=is_satlas)
	elif head == "twolinearnobias":
		return TwoLinearModel(num_feats, num_outputs, bias=False, normalize=normalize, is_satlas=is_satlas)
	elif head == "threelinearnobias":
		return ThreeLinearModel(num_feats, num_outputs, bias=False, normalize=normalize, is_satlas=is_satlas)
	elif head == "fourlinearnobias":
		return FourLinearModel(num_feats, num_outputs, bias=False, normalize=normalize, is_satlas=is_satlas)
	elif head == "mlp-linear":
		return MLPLinear(num_feats, num_outputs, is_residual=is_residual, normalize=normalize, is_satlas=is_satlas)
	else:
		raise NotImplementedError

class FourLinearModel(nn.Module):	# two separate heads for the two outputs
	def __init__(self, in_dim, out_dim, bias=True, normalize=False, is_satlas=False) -> None:
		super(FourLinearModel, self).__init__()
		self.normalize = normalize
		self.in_dim = in_dim

		self.backbone1 = nn.Sequential(	# for text
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone2 = nn.Sequential(	# for sinr
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone3 = nn.Sequential(	# for img
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone4 = nn.Sequential(	# for img
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.is_satlas = is_satlas
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		out1, out2, out3, out4 = self.backbone1(x), self.backbone2(x), self.backbone3(x), self.backbone4(x)
		if self.normalize:
			out1 = F.normalize(out1, dim=-1)
			out2 = F.normalize(out2, dim=-1)
			out3 = F.normalize(out3, dim=-1)
			out4 = F.normalize(out4, dim=-1)
		return out1, out2, out3, out4

class ThreeLinearModel(nn.Module):	# two separate heads for the two outputs
	def __init__(self, in_dim, out_dim, bias=True, normalize=False, is_satlas=False) -> None:
		super(ThreeLinearModel, self).__init__()
		self.normalize = normalize
		self.in_dim = in_dim

		self.backbone1 = nn.Sequential(	# for text
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone2 = nn.Sequential(	# for sinr
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone3 = nn.Sequential(	# for img
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.is_satlas = is_satlas
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		out1, out2, out3 = self.backbone1(x), self.backbone2(x), self.backbone3(x)
		if self.normalize:
			out1 = F.normalize(out1, dim=-1)
			out2 = F.normalize(out2, dim=-1)
			out3 = F.normalize(out3, dim=-1)
		return out1, out2, out3

class TwoLinearModel(nn.Module):	# two separate heads for the two outputs
	def __init__(self, in_dim, out_dim, bias=True, normalize=False, is_satlas=False) -> None:
		super(TwoLinearModel, self).__init__()
		self.normalize = normalize
		self.in_dim = in_dim

		self.backbone1 = nn.Sequential(
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.backbone2 = nn.Sequential(
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.is_satlas = is_satlas
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		out1, out2 = self.backbone1(x), self.backbone2(x)
		if self.normalize:
			out1 = F.normalize(out1, dim=-1)
			out2 = F.normalize(out2, dim=-1)
		return out1, out2

class LinearModel(nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, normalize=False, is_satlas=False) -> None:
		super(LinearModel, self).__init__()
		self.normalize = normalize
		self.in_dim = in_dim

		self.backbone = nn.Sequential(
			nn.Linear(in_dim,out_dim, bias=bias),
		)
		self.is_satlas = is_satlas
		logger.debug(f"self.is_satlas: {self.is_satlas}")
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		feat = self.backbone(x)
		if self.normalize:
			feat = F.normalize(feat, dim=-1)
		return feat

class MLPLinear(nn.Module):
	def __init__(self, in_dim, out_dim, is_residual=False, normalize=False, is_satlas=False) -> None:
		super(MLPLinear, self).__init__()
		self.in_dim = in_dim
		self.is_residual = is_residual

		self.backbone = nn.Sequential(
			nn.Linear(in_dim,1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024),

			nn.Linear(1024,512),
			nn.ReLU(),
			nn.BatchNorm1d(512),

			nn.Linear(512,256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			
			nn.Linear(256,in_dim),
		)
		self.head = nn.Sequential(
			nn.Linear(in_dim,out_dim),
		)
		self.normalize = normalize
		self.is_satlas = is_satlas
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		feat = self.backbone(x)
		if self.is_residual:
			feat = feat + x
		out = self.head(feat)
		if self.normalize:
			out = F.normalize(out, dim=-1)
		return out

class MLPModel(nn.Module):
	def __init__(self, in_dim, out_dim, normalize=False, is_satlas=False) -> None:
		super(MLPModel, self).__init__()
		self.in_dim = in_dim

		self.backbone = nn.Sequential(
			nn.Linear(in_dim,1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024),

			nn.Linear(1024,512),
			nn.ReLU(),
			nn.BatchNorm1d(512),

			nn.Linear(512,256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			
			nn.Linear(256,out_dim),
		)
		self.normalize = normalize
		self.is_satlas = is_satlas
		if is_satlas:
			self.layer = nn.Sequential(
				torch.nn.Conv2d(in_dim, in_dim, 3, padding=1),
				torch.nn.ReLU(inplace=True),
			)
	def forward(self, x):
		if hasattr(self,'is_satlas'):
			if self.is_satlas:
				x = self.layer(x)
				x = torch.amax(x, dim=(2,3))	# output: (batch, 128)
		feat = self.backbone(x)
		if self.normalize:
			feat = F.normalize(feat, dim=-1)
		return feat


class SingleTaskModel(nn.Module):
	""" Single-task baseline model with encoder + decoder """
	def __init__(self, backbone: nn.Module, decoder: nn.Module, num_feats:int, encoder_name=None, num_inp_feats=3):
		super(SingleTaskModel, self).__init__()
		self.num_inp_feats = num_inp_feats
		if num_inp_feats != 3:
			self.first_layer = nn.Conv2d(num_inp_feats, 3, 1) # changes from num_inp_feats channels to 3 channels
		self.backbone = backbone
		self.decoder = decoder 
		self.num_feats = num_feats
		self.encoder_name = encoder_name
		logger.debug(f"Found encoder_name:{self.encoder_name}")
		
	def forward(self, x):
		# NOTE: these lines are just for backward compatibility with currently running training
		try:
			encoder_name = self.encoder_name
		except:
			encoder_name = None

		out_size = x.size()[2:]

		if (encoder_name is not None) and (encoder_name=="graft"):
			feats = self.backbone.forward_features(x)
		elif (encoder_name is not None) and (encoder_name=="clip"):
			feats = self.backbone(x).image_embeds # B x 512
			# feats = F.normalize(feats)
		else:
			feats = self.backbone(x)
		
		if (encoder_name is not None) and (encoder_name=="taxabind"):
			feats = feats.image_embeds

		# logger.debug(f"feats: {feats.shape}")
		out = self.decoder(feats)
		# out = F.interpolate(self.decoder(feats), out_size, mode='bilinear', align_corners=True)
		return feats, out

	def embed(self, x):
		out_size = x.size()[2:]
		return self.backbone(x)

	
class SINREmbeddingModel(nn.Module):
	def __init__(self, train_params, return_logits=True):
		super(SINREmbeddingModel, self).__init__()
		self.return_logits = return_logits

		# load model
		model = sinr.models.get_model(train_params['params'], inference_only=True)
		# model.load_state_dict(train_params['state_dict'], strict=False)
		# model.eval()

		self.train_params = train_params
		self.model = model

		if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
			raster = sinr.datasets.load_env()
		else:
			raster = None
		enc = sinr.utils.CoordEncoder(train_params['params']['input_enc'], raster=raster,
								 input_dim=train_params['params']['input_dim'])
		self.enc = enc

	def forward(self, lonlat):
		normalization = torch.tensor([[1/180, 1/90]], device=lonlat.device)
		locs_enc = self.enc.encode(lonlat*normalization, normalize=False)
		if self.train_params['params']['model'] == 'HyperNet':
			out = self.model.pos_enc(locs_enc)
		else:
			out = self.model(locs_enc, return_feats=True)

		if not self.return_logits:
			return out

		if self.train_params['params']['model'] == 'HyperNet':
			logits = out @ (self.model.species_params.T)
		else:
			logits = self.model.class_emb(out)
		return logits



# Below function based on https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/utils.py#L155
class SatlasModelBackbone(torch.nn.Module):
	def __init__(self, num_inp_feats=3, fpn=True, model_name="Sentinel2_Resnet50_SI_RGB"):
		super(SatlasModelBackbone, self).__init__()
		weights_manager = satlaspretrain_models.Weights()
		self.num_inp_feats = num_inp_feats
		if num_inp_feats != 3:
			self.first = nn.Conv2d(num_inp_feats, 3, 1) # from 6 channels to 3
		if model_name == "Sentinel2_Resnet50_SI_RGB":
			model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_Resnet50_SI_RGB", fpn=False)
			model.backbone.freeze_bn = False	# NOTE: means backbone is not frozen during training

			self.backbone = model.backbone.resnet
			self.backbone_channels = 1000
		elif model_name == "Sentinel2_SwinT_SI_RGB":
			model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
			self.backbone = model.backbone.backbone
			self.backbone_channels = self.backbone.head.out_features
		elif model_name == "Sentinel2_SwinB_SI_RGB":
			model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
			self.backbone = model.backbone.backbone
			self.backbone_channels = self.backbone.head.out_features
		# self.flatten = torch.nn.Flatten(start_dim=-2, end_dim=-1)
	def forward(self, x):
		if self.num_inp_feats != 3:
			x = self.first(x)
		x = self.backbone(x)
		return x


class GRAFT(nn.Module):
    def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
        super().__init__()
        # satellite image backbone
        from transformers import CLIPVisionModelWithProjection #, CLIPTextModelWithProjection, AutoTokenizer
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
        self.patch_size = self.satellite_image_backbone.config.patch_size

        self.projector = nn.Sequential(
            nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
            nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
        )
        self.patch_size = self.satellite_image_backbone.config.patch_size
        self.norm_dim = -1

        self.temp = temp
        if temp:
            self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))

    def forward(self, image_tensor):
        # Extract features from satellite images
        # B x 197 x 768 for VIT-B/16
        hidden_state = self.satellite_image_backbone(image_tensor).last_hidden_state
        # B x 197 x 512
        satellite_image_features = F.normalize(self.projector(hidden_state), dim=self.norm_dim)
        # get the satellite image features
        return satellite_image_features

    def forward_features(self, image_tensor, normalize=True):
        # Extract features from satellite images
        # B x 512 for VIT-B/16
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        # B x 512
        if normalize:
            satellite_image_features = F.normalize(embed)
        else:
            satellite_image_features = embed
        return satellite_image_features