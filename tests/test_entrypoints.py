import importlib.util
from pathlib import Path
import subprocess,sys
import pytest


def test_root_training_refuses_historical_directory(tmp_path):
    spec=importlib.util.spec_from_file_location('quant_root_train',Path('train.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sentinel=tmp_path/'preserved.json';sentinel.write_text('unchanged')
    with pytest.raises(ValueError,match='fresh'):module.create_run(tmp_path,'missing-plan')
    assert sentinel.read_text()=='unchanged'


def test_evaluation_requires_explicit_observed_holdout_acknowledgement():
    result=subprocess.run([sys.executable,'evaluation_scripts/evaluate.py','--output','missing-study'],capture_output=True,text=True)
    assert result.returncode!=0 and '--observed-holdout' in result.stderr
