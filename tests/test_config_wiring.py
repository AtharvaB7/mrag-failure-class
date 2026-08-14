from hydra import compose, initialize_config_dir
import os

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def test_default_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
    assert cfg.model.name == "qwen2vl"
    assert cfg.retrieval.name == "hybrid"
    assert cfg.evaluation.judge_backend == "deepseek"


def test_override_model_and_retrieval_via_cli_style_args():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=llava_next", "retrieval=dense"],
        )
    assert cfg.model.name == "llava_next"
    assert cfg.model.hf_id == "llava-hf/llava-v1.6-mistral-7b-hf"
    assert cfg.retrieval.name == "dense"
    assert cfg.retrieval.dense_enabled is True
    assert cfg.retrieval.sparse_enabled is False


def test_no_retrieval_config_disables_both_retrievers():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["retrieval=no_retrieval"])
    assert cfg.retrieval.sparse_enabled is False
    assert cfg.retrieval.dense_enabled is False


def test_sparse_config_enables_only_sparse():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["retrieval=sparse"])
    assert cfg.retrieval.sparse_enabled is True
    assert cfg.retrieval.dense_enabled is False


def test_scalar_override_via_dotted_path():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["evaluation.contamination_gap_threshold=0.1"],
        )
    assert cfg.evaluation.contamination_gap_threshold == 0.1


def test_qwen2vl_config_has_expected_hf_id():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["model=qwen2vl"])
    assert cfg.model.hf_id == "Qwen/Qwen2-VL-7B-Instruct"


def test_hybrid_config_has_rrf_params():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["retrieval=hybrid"])
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.sparse_weight == 1.0
    assert cfg.retrieval.dense_weight == 1.0
