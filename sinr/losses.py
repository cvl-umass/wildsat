import torch
import utils
from torch.nn.functional import logsigmoid


def get_loss_function(params):
    if params['loss'] == 'an_full':
        return an_full
    elif params['loss'] == 'an_slds':
        return an_slds
    elif params['loss'] == 'an_ssdl':
        return an_ssdl
    elif params['loss'] == 'an_full_me':
        return an_full_me
    elif params['loss'] == 'an_slds_me':
        return an_slds_me
    elif params['loss'] == 'an_ssdl_me':
        return an_ssdl_me
    elif params['loss'] == 'pdf_count':
        return pdf_count
    elif params['loss'] == 'an_pdf':
        return an_pdf
    elif params['loss'] == 'an_multilabel':
        return an_multilabel
    elif params['loss'] == 'an_full_hypernet':
        return an_full_hypernet
    elif params['loss'] == 'an_full_hypernet_geoprior':
        return an_full_hypernet_geoprior


def neg_log(x):
    return -torch.log(x + 1e-5)

def neg_log_sig(x):
    return -torch.nn.functional.logsigmoid(x)


def bernoulli_entropy(p):
    entropy = p * neg_log(p) + (1 - p) * neg_log(1 - p)
    return entropy


def an_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id])  # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])  # entropy
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_slds(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    num_classes = loc_pred.shape[1]
    bg_class = torch.randint(low=0, high=num_classes - 1, size=(batch_size,), device=params['device'])
    bg_class[bg_class >= class_id[:batch_size]] += 1

    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred[inds[:batch_size], bg_class])  # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])  # entropy
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_full(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):]], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    # get predictions for locations and background locations
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))[:params['num_classes']]
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))[:params['num_classes']]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log(1.0 - loc_pred)  # assume negative
        loss_bg = neg_log(1.0 - loc_pred_rand)  # assume negative
    elif neg_type == 'entropy':
        loss_pos = -1 * bernoulli_entropy(1.0 - loc_pred)  # entropy
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand)  # entropy
    else:
        raise NotImplementedError
    loss_pos[inds[:batch_size], class_id] = params['pos_weight'] * neg_log(loc_pred[inds[:batch_size], class_id])

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_full_me(batch, model, params, loc_to_feats):
    return an_full(batch, model, params, loc_to_feats, neg_type='entropy')


def an_ssdl_me(batch, model, params, loc_to_feats):
    return an_ssdl(batch, model, params, loc_to_feats, neg_type='entropy')


def an_slds_me(batch, model, params, loc_to_feats):
    return an_slds(batch, model, params, loc_to_feats, neg_type='entropy')


def pdf_count(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):]], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    loc_pred = model.class_emb(loc_emb)
    loc_pred_rand = model.class_emb(loc_emb_rand)

    criterion = torch.nn.BCEWithLogitsLoss()
    # data loss
    loss_pos = criterion(loc_pred[inds[:batch_size], class_id], torch.ones(batch_size, device=params['device']))
    loss_pos += criterion(loc_pred[inds[:batch_size], -1], torch.ones(batch_size, device=params['device']))
    if neg_type == 'hard':
        loss_bg = criterion(loc_pred_rand[inds[:batch_size], class_id],
                            torch.zeros(batch_size, device=params['device']))  # assume negative
        loss_bg += criterion(loc_pred_rand[inds[:batch_size], -1],
                             torch.zeros(batch_size, device=params['device']))  # assume negative
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


C = None
pdf = None


def an_pdf_new(batch, model, params, loc_to_feats, neg_type='hard'):
    global pdf
    if pdf is None:
        pdf = utils.DataPDF(device=params['device'])
    inds = torch.arange(params['batch_size'])

    loc_feat, pos, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    pos = pos.to(params['device'])
    class_id = class_id.to(params['device'])

    batch_size = loc_feat.shape[0]
    neg_ex_mult = 2
    # create random background samples and extract features
    rand_loc = utils.rand_samples(neg_ex_mult * batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):].repeat(
                                   neg_ex_mult, 1)], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    # make sure sigmoid is included
    loc_pred = model.class_emb(loc_emb)
    loc_pred_rand = model.class_emb(loc_emb_rand)

    global C
    if C is None:
        cs = torch.load('class_counts.pt')
        cs = cs.sum() / cs
        cs = cs.to(params['device'])
        C = cs[None]

    pdf_sample = pdf.sample(pos[:, :2], pos[:, 2], loc_feat[:, -1])
    loc_log_pdf = (12.42921619684438348527348448519 + pdf_sample)
    pdf_sample_rand = pdf.sample(rand_loc, pos[:, 2].repeat(neg_ex_mult), rand_feat[:, -1])
    loc_log_pdf_rand = (12.42921619684438348527348448519 + pdf_sample_rand)
    out = torch.log(C) - logsigmoid(-loc_log_pdf[..., None]) + logsigmoid(
        loc_pred[:, :-1] - 7.6246189861593984035895533360399)
    out_rand = torch.log(C) - logsigmoid(-loc_log_pdf_rand[..., None]) + logsigmoid(
        loc_pred_rand[:, :-1] - 7.6246189861593984035895533360399)
    p0 = torch.sigmoid(out[inds[:batch_size], class_id]).detach()
    p0_rand = torch.sigmoid(out_rand[inds[:batch_size].repeat(neg_ex_mult), class_id.repeat(neg_ex_mult)]).detach()
    pdf_sample = 500.0 * 500.0 / 4.0 * pdf_sample
    pdf_sample_rand = 500.0 * 500.0 / 4.0 * pdf_sample_rand
    weight = (-1.0 * p0 + C[0, class_id] * (pdf_sample + 0.25)) \
             * (2047 * p0 + C[0, class_id] * (pdf_sample + 0.25)) \
             / (2048 * ((C[0, class_id] * (pdf_sample + 0.25)) ** 2)) \
             * ((torch.exp(out[inds[:batch_size], class_id].detach()) + 1.0) / torch.sigmoid(
        -out[inds[:batch_size], class_id].detach()))
    weight_rand = (-1.0 * p0_rand + C[0, class_id.repeat(neg_ex_mult)] * (pdf_sample_rand + 0.25)) \
                  * (2047 * p0_rand + C[0, class_id.repeat(neg_ex_mult)] * (pdf_sample_rand + 0.25)) \
                  / (2048 * ((C[0, class_id.repeat(neg_ex_mult)] * (pdf_sample_rand + 0.25)) ** 2)) \
                  * ((torch.exp(
        out_rand[inds[:batch_size].repeat(neg_ex_mult), class_id.repeat(neg_ex_mult)].detach()) + 1.0) / torch.sigmoid(
        -out_rand[inds[:batch_size].repeat(neg_ex_mult), class_id.repeat(neg_ex_mult)].detach()))
    weight = 1+0*weight.clamp(0,100)
    weight_rand = 1+0*weight_rand.clamp(0,100)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    # data loss
    loss_pos = (weight * criterion(out[inds[:batch_size], class_id],
                                                         torch.full((batch_size,), 1.0,
                                                                    device=params['device']))).mean()
    if neg_type == 'hard':
        loss_bg = (weight_rand * criterion(
            out_rand[torch.arange(neg_ex_mult * class_id.shape[0]), class_id.repeat(neg_ex_mult)],
            0.001+torch.zeros(neg_ex_mult * batch_size, device=params['device']))).mean()  # assume negative
        # loss_bg += criterion(-15.0-out_rand[torch.arange(neg_ex_mult*class_id.shape[0]), class_id.repeat(neg_ex_mult)],
        #                    torch.zeros(neg_ex_mult*batch_size, device=params['device']))  # assume negative
        # loss_bg += criterion(loc_pred_rand[:, -1],
        #                    torch.zeros(neg_ex_mult*batch_size, device=params['device']))  # assume negative
        # loss_bg += criterion(-15.0-loc_pred_rand[:, -1],
        #                    torch.zeros(neg_ex_mult*batch_size, device=params['device']))  # assume negative
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos + loss_bg

    return loss


def an_pdf(batch, model, params, loc_to_feats, neg_type='hard'):
    global pdf
    if pdf is None:
        pdf = utils.DataPDFH3(device=params['device'])
    inds = torch.arange(params['batch_size'])

    loc_feat, pos, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    batch_size = loc_feat.shape[0]
    neg_ex_mult = 8
    # create random background samples and extract features
    rand_loc = utils.rand_samples(neg_ex_mult * batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):].repeat(
                                   neg_ex_mult, 1)], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    # make sure sigmoid is included
    loc_pred = model.class_emb(loc_emb)
    loc_pred_rand = model.class_emb(loc_emb_rand)

    global C
    if C is None:
        cs = torch.load('class_counts.pt')
        cs = cs.sum() / cs
        cs = cs.to(params['device'])
        C = cs[None]

    log_bincount = 14.51704347970842527157983497684
    pdf_sample = pdf.sample_log(pos[:, :2], pos[:, 2], loc_feat[:, -1].cpu()).to(params['device'])
    loc_log_pdf = (log_bincount + pdf_sample)
    pdf_sample_rand = pdf.sample_log(rand_loc.cpu(), pos[:, 2].repeat(neg_ex_mult), rand_feat[:, -1].cpu()).to(params['device'])
    loc_log_pdf_rand = (log_bincount + pdf_sample_rand)
    out = torch.log(C) - logsigmoid(-loc_log_pdf[..., None]) + logsigmoid(
        loc_pred[:, :-1] - 7.6246189861593984035895533360399)
    out_rand = torch.log(C) - logsigmoid(-loc_log_pdf_rand[..., None]) + logsigmoid(
        loc_pred_rand[:, :-1] - 7.6246189861593984035895533360399)

    criterion = torch.nn.BCEWithLogitsLoss()
    # data loss
    loss_pos = criterion(out[inds[:batch_size], class_id], torch.full((batch_size,), 1.0, device=params['device']))
    if neg_type == 'hard':
        loss_bg = criterion(out_rand[torch.arange(neg_ex_mult * class_id.shape[0]), class_id.repeat(neg_ex_mult)],
                            torch.zeros(neg_ex_mult * batch_size, device=params['device']))  # assume negative
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()# + loss_extra.mean()

    return loss


def an_multilabel(batch, model, params, loc_to_feats, neg_type='hard'):
    loc_feat, _, labels = batch
    loc_feat = loc_feat.to(params['device'])
    labels = labels.to(params['device'])

    assert model.inc_bias == False

    # get location embeddings
    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = model.class_emb(loc_emb)

    # data loss
    loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(loc_pred, labels)

    # total loss
    loss = loss_pos.mean()

    return loss


def an_full_hypernet(batch, model, params, loc_to_feats, neg_type='hard', class_samples=192):
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    class_id_real = torch.randint(0,params['num_classes']-1, size=(batch_size, class_samples-1), device=params['device'])
    class_id_real[class_id_real >= class_id[:,None]] += 1
    class_id_real = torch.cat([class_id[:,None], class_id_real], dim=1)
    class_id_fake = torch.randint(0,params['num_classes'], size=(batch_size, class_samples), device=params['device'])
    class_id_cat = torch.cat([class_id_real, class_id_fake], 0)

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):]], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, class_id_cat)
    loc_pred = loc_emb_cat[:batch_size, :]
    loc_pred_rand = loc_emb_cat[batch_size:, :]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log_sig(-loc_pred)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg = neg_log_sig(-loc_pred_rand)*(params['num_classes']/class_samples)  # assume negative
    else:
        raise NotImplementedError
    loss_pos[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred[:, 0])

    # total loss
    loss = loss_pos.sum()/params['num_classes']/loss_pos.shape[0] + loss_bg.sum()/params['num_classes']/loss_bg.shape[0]

    return loss


def an_full_hypernet_geoprior(batch, model, params, loc_to_feats, neg_type='hard', class_samples=192):
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    class_id_real = torch.randint(0,params['num_classes']-1, size=(batch_size, class_samples-1), device=params['device'])
    class_id_real[class_id_real >= class_id[:,None]] += 1   # RN NOTE: What's this for? Is it to make sure none of them are the same as current class?
    class_id_real = torch.cat([class_id[:,None], class_id_real], dim=1)
    class_id_fake = torch.randint(0,params['num_classes'], size=(batch_size, class_samples), device=params['device'])
    class_id_cat = torch.cat([class_id_real, class_id_fake], 0)

    # create random background samples and extract features
    # RN NOTE: doesn't check against current location?
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):]], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat, loc_emb_cat2, species_sim = model(loc_cat, class_id_cat)
    s1, s2, uniq = species_sim
    s1 = s1/torch.norm(s1, dim=1, keepdim=True)
    s2 = s2/torch.norm(s2, dim=1, keepdim=True)
    imap = torch.zeros(uniq.max()+1, dtype=int, device=uniq.device)
    imap[uniq] = torch.arange(len(uniq), device=uniq.device)
    loc_pred = loc_emb_cat[:batch_size, :]
    loc_pred_rand = loc_emb_cat[batch_size:, :]
    loc_pred2 = loc_emb_cat2[:batch_size, :]
    loc_pred_rand2 = loc_emb_cat2[batch_size:, :]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log_sig(-loc_pred)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg = neg_log_sig(-loc_pred_rand)*(params['num_classes']/class_samples)  # assume negative
        loss_pos2 = neg_log_sig(-loc_pred2)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg2 = neg_log_sig(-loc_pred_rand2)*(params['num_classes']/class_samples)  # assume negative

        loss3 = 0
        for x,y in [(s1, s2), (s2, s1)]:    # RN NOTE: this is between embedding from text and learned species embeddings
            l3 = (x[imap[class_id_real[:,:1]]] * y[imap[class_id_real]]).sum(dim=-1)
            l3[:,0] *= -1
            l3 = neg_log_sig(-params['geoprior_temp']*l3)
            loss3 += 0.5*(l3[:,0].mean() + l3[:,1:].mean())
    else:
        raise NotImplementedError
    loss_pos[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred[:, 0])
    loss_pos2[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred2[:, 0])
    # total loss
    loss = loss_pos.sum()/params['num_classes']/loss_pos.shape[0] + loss_bg.sum()/params['num_classes']/loss_bg.shape[0]
    loss2 = loss_pos2.sum()/params['num_classes']/loss_pos2.shape[0] + loss_bg2.sum()/params['num_classes']/loss_bg2.shape[0]

    return loss+loss2+loss3