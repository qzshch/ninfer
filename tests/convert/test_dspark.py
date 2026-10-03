from copy import deepcopy
import pytest
from tools.convert.qwen3_5 import draft_config
from tools.convert.dspark import dspark_config

def configs():
    target = {"hidden_size":5120,"vocab_size":248320,"num_hidden_layers":64,
              "max_position_embeddings":262144}
    layer = {"hidden_size":5120,"vocab_size":248320,"num_hidden_layers":5,
        "intermediate_size":17408,"num_attention_heads":20,"num_key_value_heads":4,
        "head_dim":256,"hidden_act":"silu","rms_norm_eps":1e-6,
        "max_position_embeddings":262144,"sliding_window":2048,
        "layer_types":["sliding_attention"]*5,
        "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":1.0,
                           "mrope_section":[44,44,40]}}
    raw={"architectures":["DSparkDraftModel"],"speculators_model_type":"dspark",
         "markov_rank":256,"markov_head_type":"vanilla","sample_from_anchor":True,
         "sliding_window_non_causal":False,"block_size":8,"draft_vocab_size":248320,
         "aux_hidden_state_layer_ids":[4,12,20,28,36,44,52,60],"mask_token_id":248077,
         "transformer_layer_config":layer}
    return raw,target

def test_official_aux_ids_are_decoder_outputs_not_embedding_indices():
    raw,target=configs(); before=deepcopy(raw)
    config=draft_config(raw,target,"dspark")
    assert config["dflash_config"]["target_layer_ids"]==[3,11,19,27,35,43,51,59]
    assert config["architectures"]==["DSparkDraftModel"]
    assert config["dspark_config"]["sample_from_anchor"] is True
    assert config["dspark_config"]["sliding_window_non_causal"] is False
    assert raw==before

@pytest.mark.parametrize("field,value",[("sample_from_anchor",False),("markov_rank",128),
    ("sliding_window_non_causal",True),("block_size",16),("draft_vocab_size",100),
    ("aux_hidden_state_layer_ids",[0,4]),("aux_hidden_state_layer_ids",[64,65])])
def test_reject_unqualified_math_or_tap_domains(field,value):
    raw,target=configs();raw[field]=value
    with pytest.raises(ValueError):dspark_config(raw,target)

@pytest.mark.parametrize("field,value",[("head_dim",128),("sliding_window",4096),
    ("num_attention_heads",40),("hidden_size",2560)])
def test_reject_other_geometries(field,value):
    raw,target=configs();raw["transformer_layer_config"][field]=value
    with pytest.raises(ValueError):dspark_config(raw,target)

def test_confidence_requires_the_official_markov_input():
    raw,target=configs()
    raw.update(enable_confidence_head=True, confidence_head_with_markov=True)
    assert dspark_config(raw,target)["dspark_config"]["confidence_head"] is True
    raw["confidence_head_with_markov"]=False
    with pytest.raises(ValueError,match="predecessor embedding"):dspark_config(raw,target)
