import torch

# Used from RLPYT
def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])



def spr_loss(states_k,actions_k,K=5):
    yk_q = None # Todo calculate by prediction head q(g_o(z))
    yk_gm = None # Todo calculate by target projection head g_m(z_)

    # Prediction Loss (cosine similarities)
    loss_spr = - torch.sum(torch.cosine_similarity(yk_q,yk_gm))



def rl_loss(qs,idx,samples_replay):
    pass

