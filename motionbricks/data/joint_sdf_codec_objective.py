"""Native MB source-code reconstruction, separate from causal actor generation."""
import torch

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


def native_codec_reconstruction(model,result):
    """Retain the meaning of source codes while adapting the decoder.

    This is the tokenizer's supervised autoencoder path. Source Root conditions
    are confined to this auxiliary reconstruction, never the actor's forward
    conditions or generated motion. Its boundary history is the source history,
    not a different generated-history trajectory used by the actor objective.
    """
    if model.codec.encoder.training or model.codec.quantizer.training:
        raise ValueError('This adaptation requires a frozen evaluation-mode vocabulary')
    target=dict(result['target']);local=target['local_motions'].detach()
    with torch.no_grad():
        root=extract_feature_from_motion_rep(local,model.codec.motion_rep,model.codec.decoder_external_cond_feature_mode)
        mask=torch.zeros(local.shape[:2],dtype=torch.bool,device=local.device);mask[:,:4]=True
        target['executed_history_positions']=model.inference.local_motion_rep.inverse(local,True,
            return_all=True,joint_positions_from='global_rot_data')['posed_joints'][:,:4].detach()
    output=model.codec(local,target_cond=local,has_target_cond=mask,external_cond=root.detach())
    return model.loss_model.loss(target,output,None)
