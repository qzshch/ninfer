"""Explicit RedHat DSpark anchor-block configuration (fixed greedy window)."""
from copy import deepcopy


def dspark_config(raw, target):
    from .qwen3_5 import draft_config
    expected = {'architectures': ['DSparkDraftModel'], 'speculators_model_type': 'dspark',
                'markov_rank': 256, 'markov_head_type': 'vanilla',
                'sample_from_anchor': True, 'block_size': 8,
                'sliding_window_non_causal': False}
    for key, value in expected.items():
        if raw.get(key) != value or type(raw.get(key)) is not type(value):
            raise ValueError(f'dspark: unsupported or missing {key}')
    layer = deepcopy(raw['transformer_layer_config'])
    if layer.get('hidden_size') != target['hidden_size'] or layer.get('vocab_size') != target['vocab_size']:
        raise ValueError('dspark: target hidden/vocabulary geometry mismatch')
    if raw.get('draft_vocab_size') != target['vocab_size']:
        raise ValueError('dspark: vocabulary remapping is not supported')
    taps = raw.get('aux_hidden_state_layer_ids')
    if not isinstance(taps, list) or not taps or any(type(i) is not int or i < 1 for i in taps):
        raise ValueError('dspark: explicit post-layer auxiliary indices are required')
    # Speculators/vLLM aux IDs include the embedding state at index zero.
    # NInfer captures zero-based decoder outputs; official update_dspark uses i-1.
    layer.update(architectures=['DFlashDraftModel'], is_causal=False,
                 num_target_layers=target['num_hidden_layers'],
                 dflash_config={'target_layer_ids': [i-1 for i in taps],
                                'mask_token_id': raw['mask_token_id']})
    result = draft_config(layer, target, 'dflash')
    if (result['head_dim'], result['num_attention_heads'], result['num_key_value_heads'],
        result.get('sliding_window')) != (256, 20, 4, 2048):
        raise ValueError('dspark: only D256/Q20/KV4/window2048 is qualified')
    if any(kind != 'sliding_attention' for kind in result['layer_types']):
        raise ValueError('dspark: all-local attention is required')
    result['architectures'] = ['DSparkDraftModel']
    result['dspark_config'] = {k: raw[k] for k in ('markov_rank', 'markov_head_type',
        'sample_from_anchor', 'block_size', 'sliding_window_non_causal')}
    result['dspark_config']['confidence_head'] = bool(raw.get('enable_confidence_head', False))
    if result['dspark_config']['confidence_head'] and not raw.get('confidence_head_with_markov'):
        raise ValueError('dspark: confidence head must include predecessor embedding')
    return result
