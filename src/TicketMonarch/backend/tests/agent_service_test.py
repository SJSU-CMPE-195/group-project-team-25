import types
import numpy as np
import pytest

from TicketMonarch.backend import agent_service


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.array(arr, dtype=float)

    def float(self):
        return self

    def unsqueeze(self, dim):
        return self

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def squeeze(self, dim=None):
        if dim is None:
            return FakeTensor(np.squeeze(self.arr))
        return FakeTensor(np.squeeze(self.arr, axis=dim))

    def item(self):
        return float(np.array(self.arr).reshape(-1)[0])

    def norm(self):
        return FakeScalar(np.linalg.norm(self.arr))

    def __getitem__(self, idx):
        return FakeTensor(self.arr[idx])

    def tolist(self):
        return self.arr.tolist()


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyBuffer:
    def __init__(self):
        self.pushed = []
        self.gae_called = False
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def push(self, *args, **kwargs):
        self.pushed.append((args, kwargs))

    def compute_gae(self, **kwargs):
        self.gae_called = True


class DummyNetwork:
    def __init__(self, logits_seq=None, values_seq=None):
        self.logits_seq = logits_seq or [
            [0.9, 0.1, 0, 0, 0, 0, 0],
            [0, 0, 0.1, 0.2, 0.3, 0.9, 0.4],
        ]
        self.values_seq = values_seq or [0.25, 0.75]
        self.calls = 0
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True

    def train(self):
        self.train_called = True

    def __call__(self, obs_t, hidden, action_mask=None):
        i = min(self.calls, len(self.logits_seq) - 1)
        self.calls += 1
        logits = FakeTensor([self.logits_seq[i]])
        values = FakeTensor([self.values_seq[i]])
        new_hidden = (FakeTensor([[[1.0, 2.0]]]), FakeTensor([[[3.0, 4.0]]]))
        return logits, values, new_hidden


class DummyAgent:
    def __init__(self):
        self.device = "cpu"
        self.network = DummyNetwork()
        self.buffer = DummyBuffer()
        self.optimizer = types.SimpleNamespace(param_groups=[{"lr": 1.0}])
        self.config = types.SimpleNamespace(
            lr=1.0, num_epochs=5, gamma=0.99, gae_lambda=0.95
        )
        self._hidden = (FakeTensor([[[0.0, 0.0]]]), FakeTensor([[[0.0, 0.0]]]))
        self.saved_path = None

    def load(self, checkpoint_path):
        pass

    def reset_hidden(self):
        self._hidden = (FakeTensor([[[0.0, 0.0]]]), FakeTensor([[[0.0, 0.0]]]))

    def get_hidden(self):
        return self._hidden

    def select_action(self, obs, action_mask=None):
        return 5, -0.2, 0.3

    def update(self):
        return {"policy_loss": 0.1, "value_loss": 0.2}

    def save(self, checkpoint_path):
        self.saved_path = checkpoint_path


def test_create_agent_invalid_algorithm():
    with pytest.raises(ValueError, match="Unknown RL algorithm"):
        agent_service.AgentService(algorithm="bad_algo")


def test_get_agent_service_singleton(monkeypatch):
    dummy = object()
    monkeypatch.setattr(agent_service, "_agent_service", None)
    monkeypatch.setattr(agent_service, "AgentService", lambda: dummy)

    a = agent_service.get_agent_service()
    b = agent_service.get_agent_service()

    assert a is dummy
    assert b is dummy


def test_evaluate_session_no_checkpoint(monkeypatch):
    monkeypatch.setattr(
        agent_service.AgentService, "_create_agent", lambda self: DummyAgent()
    )

    svc = agent_service.AgentService(checkpoint_path="does-not-matter", algorithm="ppo")
    svc._loaded = False

    result = svc.evaluate_session(session=None)

    assert result["decision"] == "allow"
    assert result["reason"] == "no_checkpoint"


def test_rolling_evaluate_no_checkpoint(monkeypatch):
    monkeypatch.setattr(
        agent_service.AgentService, "_create_agent", lambda self: DummyAgent()
    )

    svc = agent_service.AgentService(checkpoint_path="does-not-matter", algorithm="ppo")
    svc._loaded = False

    result = svc.rolling_evaluate(session=None)

    assert result["bot_probability"] == 0.0
    assert result["deploy_honeypot"] is False


def test_online_learn_no_checkpoint(monkeypatch):
    monkeypatch.setattr(
        agent_service.AgentService, "_create_agent", lambda self: DummyAgent()
    )

    svc = agent_service.AgentService(checkpoint_path="does-not-matter", algorithm="ppo")
    svc._loaded = False

    result = svc.online_learn(session=None, true_label=1)

    assert result == {"updated": False, "reason": "no_checkpoint"}


def make_service(monkeypatch, algorithm="ppo", patch_create_agent=True):
    if patch_create_agent:
        monkeypatch.setattr(
            agent_service.AgentService, "_create_agent", lambda self: DummyAgent()
        )
    monkeypatch.setattr(agent_service.Path, "exists", lambda self: False)
    return agent_service.AgentService(checkpoint_path="dummy-path", algorithm=algorithm)


def test_create_agent_branch_dg(monkeypatch):
    monkeypatch.setattr(
        agent_service, "DGLSTM", lambda config, **kwargs: ("dg", kwargs)
    )
    monkeypatch.setattr(agent_service, "DGConfig", lambda: "dgcfg")
    svc = make_service(monkeypatch, algorithm="dg", patch_create_agent=False)
    svc.env_config = types.SimpleNamespace(event_dim=9)
    assert svc._create_agent()[0] == "dg"


def test_create_agent_branch_soft_ppo(monkeypatch):
    monkeypatch.setattr(
        agent_service, "SoftPPOLSTM", lambda config, **kwargs: ("soft", kwargs)
    )
    monkeypatch.setattr(agent_service, "SoftPPOConfig", lambda: "softcfg")
    svc = make_service(monkeypatch, algorithm="soft_ppo", patch_create_agent=False)
    svc.env_config = types.SimpleNamespace(event_dim=9)
    assert svc._create_agent()[0] == "soft"


def test_build_windows_truncates(monkeypatch):
    svc = make_service(monkeypatch)
    svc.env_config.window_size = 4
    svc.env_config.min_events = 2
    svc.env_config.max_windows = 2

    class DummyEncoder:
        def __init__(self, cfg):
            pass

        def build_timeline(self, session):
            return list(range(10))

    monkeypatch.setattr(agent_service, "EventEncoder", DummyEncoder)

    timeline, windows = svc._build_windows(session=None)

    assert len(timeline) == 10
    assert len(windows) == 2


def install_fake_torch(monkeypatch):
    monkeypatch.setattr(agent_service.torch, "no_grad", FakeNoGrad)
    monkeypatch.setattr(agent_service.torch, "from_numpy", lambda arr: FakeTensor(arr))
    monkeypatch.setattr(
        agent_service.F,
        "softmax",
        lambda logits, dim=-1: logits,
    )


def test_evaluate_session_with_windows(monkeypatch):
    svc = make_service(monkeypatch)
    svc._loaded = True

    install_fake_torch(monkeypatch)

    class DummyEncoder:
        def __init__(self, cfg):
            pass

        def build_timeline(self, session):
            return [1, 2, 3, 4]

        def encode_window(self, window):
            return np.array([0.1, 0.2])

    monkeypatch.setattr(agent_service, "EventEncoder", DummyEncoder)
    monkeypatch.setattr(
        svc,
        "_build_windows",
        lambda session: ([1, 2, 3, 4], [[1, 2], [3, 4]]),
    )

    result = svc._evaluate_session(session=None)

    assert result["decision"] == "allow"
    assert result["num_windows"] == 2
    assert result["windows_processed"] == 2
    assert len(result["action_history"]) == 2


def test_rolling_evaluate_with_honeypot(monkeypatch):
    svc = make_service(monkeypatch)
    svc._loaded = True
    svc.agent.network = DummyNetwork(
        logits_seq=[
            [0.1, 0.9, 0, 0, 0, 0, 0],  # non-final -> honeypot
            [0, 0, 0.1, 0.2, 0.3, 0.1, 0.4],  # final
        ],
        values_seq=[0.1, 0.2],
    )

    install_fake_torch(monkeypatch)

    class DummyEncoder:
        def __init__(self, cfg):
            pass

        def encode_window(self, window):
            return np.array([0.1, 0.2])

    monkeypatch.setattr(agent_service, "EventEncoder", DummyEncoder)
    monkeypatch.setattr(
        svc,
        "_build_windows",
        lambda session: ([1, 2, 3, 4], [[1, 2], [3, 4]]),
    )

    result = svc._rolling_evaluate(session=None)

    assert result["deploy_honeypot"] is True
    assert result["num_windows"] == 2
    assert "action_distribution" in result


def test_online_learn_success(monkeypatch):
    svc = make_service(monkeypatch)
    svc._loaded = True

    class DummyEncoder:
        def __init__(self, cfg):
            pass

        def encode_window(self, window):
            return np.array([0.1, 0.2])

    monkeypatch.setattr(agent_service, "EventEncoder", DummyEncoder)
    monkeypatch.setattr(
        svc,
        "_build_windows",
        lambda session: ([1, 2, 3, 4], [[1, 2], [3, 4]]),
    )

    evaluations = [
        {"decision": "block", "p_allow": 0.2, "p_suspicious": 0.8},
        {"decision": "allow", "p_allow": 0.9, "p_suspicious": 0.1},
    ]
    monkeypatch.setattr(svc, "_evaluate_session", lambda session: evaluations.pop(0))
    monkeypatch.setattr(
        agent_service,
        "compute_terminal_reward",
        lambda cfg, action, true_label, meta, rng: (1.0, {}),
    )

    session = types.SimpleNamespace(metadata={})
    result = svc._online_learn(session=session, true_label=1)

    assert result["updated"] is True
    assert result["steps"] == 2
    assert result["improvement"] == "IMPROVED"
    assert svc.agent.buffer.gae_called is True
    assert svc.agent.saved_path == "dummy-path"


def test_get_hidden_state_info(monkeypatch):
    svc = make_service(monkeypatch)

    result = svc.get_hidden_state_info()

    assert "lstm_hidden_norm" in result
    assert "lstm_cell_norm" in result
    assert isinstance(result["lstm_hidden_values"], list)
