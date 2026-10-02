from copy import deepcopy
import os
import pytest
from tools.artifact.reader import Artifact
from tools.artifact.writer import ArtifactWriter
from tools.artifact.schema import ArtifactError,TensorSpec,ResourceSpec
from tools.convert.attach_dspark import compose_plan

def fixture(path,donor=False,tokenizer=b"tokens"):
    specs=[TensorSpec("w",(2,4),"bf16","contiguous_le_v1"),
        TensorSpec("scale",(),"fp32","contiguous_le_v1"),ResourceSpec("tokens",len(tokenizer))]
    components={"text":{"config":{"hidden_size":2},"resources":{"tokenizer.json":"tokens"}}}
    bindings={"text/output_head":{"object":"w"}}
    uses=[{"parameter":"text/output_head","input":"text/final_hidden",
        "activation_policy":"AllowA4","auxiliaries":{"activation_input_divisor":{"object":"scale"}}}]
    if donor:
        components["dspark"]={"target":"text","config":{}}
        bindings["dspark/query"]={"parts":[{"object":"w","range":[4,8]},{"object":"w","range":[0,4]}]}
        uses += [{"parameter":"dspark/query","input":"dspark/hidden","activation_policy":"A16Only"}]
    with ArtifactWriter(path,specs,components=components,bindings=bindings,uses=uses) as w:
        w.write_object("w",bytes([9 if donor else 7])*16);w.write_object("scale",bytes([5 if donor else 2])*4);w.write_object("tokens",tokenizer)

def test_encoded_plan_keeps_base_head_policy_ranges_and_target_bytes(tmp_path):
    b,d,o=[tmp_path/(n+".ninfer") for n in ("base","donor","out")];fixture(b);fixture(d,True)
    with Artifact(b) as base,Artifact(d) as donor:
        specs,components,bindings,uses,copies=compose_plan(base,donor)
        with ArtifactWriter(o,specs,components=components,bindings=bindings,uses=uses) as w:
            for a,old,new in copies:w.write_object(new,a.read_object(old))
        with Artifact(o) as output:
            for obj in base.objects:assert output.read_object(obj.id)==base.read_object(obj.id)
            assert bindings["text/output_head"]==base.directory.bindings["text/output_head"]
            assert bindings["dspark/query"]["parts"][0]["range"]==[4,8]
            shared=next(u for u in uses if u["input"]=="dspark/final_hidden")
            assert shared=={**deepcopy(base.directory.uses[0]),"input":"dspark/final_hidden"}
            assert output.read_object("import/dspark/w")==donor.read_object("w")

def test_reject_different_tokenizer(tmp_path):
    b,d=[tmp_path/(n+".ninfer") for n in ("base","donor")];fixture(b);fixture(d,True,b"changed")
    with Artifact(b) as base,Artifact(d) as donor:
        with pytest.raises(ArtifactError,match="resource differs"):compose_plan(base,donor)

class RenameWriter(ArtifactWriter):
    def _publish_completed_file(self,source,target):
        # Test a publication provider that consumes its completed temporary file.
        if target.exists():raise FileExistsError(target)
        os.rename(source,target)

def test_publisher_may_consume_temp_and_existing_destination_survives(tmp_path):
    p=tmp_path/"published.ninfer"
    with RenameWriter(p,[ResourceSpec("r",1)],components={"text":{"config":{"hidden_size":1}}},bindings={},uses=[]) as w:w.write_object("r",b"a")
    before=p.read_bytes()
    with Artifact(p) as r:assert r.read_object("r")==b"a"
    with pytest.raises(FileExistsError):
        with RenameWriter(p,[ResourceSpec("r",1)],components={"text":{"config":{"hidden_size":1}}},bindings={},uses=[]) as w:w.write_object("r",b"b")
    assert p.read_bytes()==before
