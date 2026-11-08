"""Test multi-backend LLM analysis."""
import json
import urllib.request
from unittest.mock import MagicMock, patch

from predictive_agent.llm import LLMAnalyzer, LLMBackend


def test_llm_backend_enum():
    """Test LLM backend enum."""
    assert LLMBackend.OLLAMA.value == "ollama"
    assert LLMBackend.SAIA.value == "saia"
    assert LLMBackend.TUD.value == "tud"
    assert LLMBackend.OPENAI.value == "openai"


def test_llm_analyzer_creation_ollama():
    """Test creating Ollama analyzer."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://ollama.llm.svc.cluster.local:11434",
        model="qwen3-30b-a3b:latest",
    )
    assert analyzer.backend == LLMBackend.OLLAMA
    assert analyzer.url == "http://ollama.llm.svc.cluster.local:11434"
    assert analyzer.model == "qwen3-30b-a3b:latest"


def test_llm_analyzer_creation_saia():
    """Test creating SAIA analyzer."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.SAIA,
        url="https://chat-ai.academiccloud.de/v1",
        model="qwen3.5-35b-a3b",
        api_key="test-key",
    )
    assert analyzer.backend == LLMBackend.SAIA
    assert analyzer.api_key == "test-key"


def test_llm_analyzer_creation_tud():
    """Test creating TUD analyzer."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.TUD,
        url="https://llm-service.ai.tu-darmstadt.de/v1",
        model="GLM-5.2-AWQ-INT4",
        api_key="test-key",
    )
    assert analyzer.backend == LLMBackend.TUD


def test_llm_analyzer_creation_openai():
    """Test creating OpenAI analyzer."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="test-key",
    )
    assert analyzer.backend == LLMBackend.OPENAI


def test_llm_build_prompt_ollama():
    """Test prompt building for Ollama."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test-model",
    )
    prompt = analyzer.build_prompt(
        issue="Pod crash",
        context="ns=test pod=app-0 status=CrashLoopBackOff",
        prediction={"risk_score": 0.85, "ttf_minutes": 12, "confidence": 0.87}
    )
    assert "Pod crash" in prompt
    assert "CrashLoopBackOff" in prompt
    assert "0.85" in prompt
    assert "12" in prompt


def test_llm_build_prompt_with_prediction():
    """Test prompt includes prediction data."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.SAIA,
        url="https://chat-ai.academiccloud.de/v1",
        model="test",
        api_key="key",
    )
    prediction = {
        "risk_score": 0.92,
        "ttf_minutes": 8,
        "confidence": 0.91,
        "markov_state": "CRITICAL",
        "memory_trend": 2.5,
    }
    prompt = analyzer.build_prompt(
        issue="OOM predicted",
        context="memory rising at 2.5 MiB/min",
        prediction=prediction,
    )
    assert "0.92" in prompt
    assert "CRITICAL" in prompt
    assert "2.5" in prompt


def test_llm_parse_response():
    """Test parsing LLM JSON response."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test",
    )
    response = '{"analysis": "Memory leak in container", "severity": "high", "action": "Restart pod", "command": "kubectl delete pod"}'
    parsed = analyzer.parse_response(response)
    assert parsed["analysis"] == "Memory leak in container"
    assert parsed["severity"] == "high"


def test_llm_parse_response_invalid():
    """Test parsing invalid JSON response."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test",
    )
    parsed = analyzer.parse_response("not json at all")
    assert parsed is not None
    assert "analysis" in parsed  # Should have fallback


def test_llm_parse_response_fenced_json():
    """Parse JSON inside markdown code fences (common with reasoning models)."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test",
    )
    fenced = '```json\n{"analysis": "memory leak", "severity": "high", "action": "Restart", "command": "kubectl delete pod"}\n```'
    parsed = analyzer.parse_response(fenced)
    assert parsed["severity"] == "high"
    assert parsed["action"] == "Restart"


def test_llm_parse_response_prose_wrapped():
    """Extract JSON block even when surrounded by prose."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test",
    )
    prose = ('Here is my analysis:\n'
             '{"analysis": "cpu burn", "severity": "medium", "action": "Inspect", "command": "kubectl logs app"}\n'
             'Let me know if you need more.')
    parsed = analyzer.parse_response(prose)
    assert parsed["action"] == "Inspect"


def test_llm_build_prompt_requires_json_only():
    """Prompt must instruct the model to answer with JSON only."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
    )
    prompt = analyzer.build_prompt("Pod crash", "ns=test")
    assert "JSON object" in prompt


def test_llm_max_tokens_configurable():
    """Analyzer accepts a configurable max_tokens (default must clear the
    reasoning-model budget so content is never swallowed by reasoning)."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
        max_tokens=2048,
    )
    assert analyzer.max_tokens == 2048
    assert LLMAnalyzer(
        backend=LLMBackend.OPENAI, url="u", model="m"
    ).max_tokens == 4096


def test_llm_openai_payload_includes_max_tokens():
    """Regression: omit max_tokens and LiteLLM requests the model's full
    output limit (204800) -> ContextWindowExceededError -> 400 on every call.
    The OpenAI-compatible payload must always set a bounded max_tokens."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
        api_key="k",
    )
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        mock = MagicMock()
        mock.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"analysis": "a", "severity": "low", "action": "Monitor", "command": ""}'}}]
        }).encode()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=None)
        return mock

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        analyzer.analyze("x", "y")
    assert "max_tokens" in captured["body"]
    assert captured["body"]["max_tokens"] == analyzer.max_tokens


def test_llm_analyze_http_error_action_is_honest():
    """An HTTP rejection (400/401/...) is a config problem, not connectivity."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
    )
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "http://localhost:4000/v1/chat/completions", 400, "Bad Request", {}, None)):
        result = analyzer.analyze("Pod crash", "ns=test")
    assert "Check LLM connectivity" not in result["action"]
    assert "400" in result["analysis"]


def test_llm_analyze_connection_error_keeps_connectivity_action():
    """True connectivity failures keep the connectivity action."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
    )
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        result = analyzer.analyze("Pod crash", "ns=test")
    assert result["action"] == "Check LLM connectivity"


def test_llm_analyze_timeout_action_is_honest():
    """A timeout is not a connectivity failure; don't mislabel it."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OPENAI,
        url="http://localhost:4000/v1",
        model="m",
        timeout=5,
    )
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = analyzer.analyze("Pod crash", "ns=test")
    assert "Check LLM connectivity" not in result["action"]


# ─── Enhanced tests: actual API calls (mocked) ─────────────────────────────


def test_llm_analyze_ollama_mock():
    """Test analyze() makes HTTP request to Ollama API."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test-model",
    )
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "response": '{"analysis": "OOM risk", "severity": "high", "action": "Restart", "command": "kubectl delete pod"}'
    }).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = analyzer.analyze("Pod crash", "ns=test")
        assert result is not None
        assert "analysis" in result


def test_llm_analyze_saia_mock():
    """Test analyze() makes HTTP request to SAIA API (OpenAI-compatible)."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.SAIA,
        url="https://chat-ai.academiccloud.de/v1",
        model="test-model",
        api_key="test-key",
    )
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"analysis": "Memory leak", "severity": "medium", "action": "Monitor", "command": ""}'}}]
    }).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = analyzer.analyze("High memory", "ns=test")
        assert result is not None
        assert "analysis" in result


def test_llm_analyze_tud_mock():
    """Test analyze() makes HTTP request to TUD API."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.TUD,
        url="https://llm-service.ai.tu-darmstadt.de/v1",
        model="test-model",
        api_key="test-key",
    )
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"analysis": "CPU throttle", "severity": "low", "action": "Scale", "command": ""}'}}]
    }).encode()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = analyzer.analyze("CPU high", "ns=test")
        assert result is not None
        assert "analysis" in result


def test_llm_analyze_error_handling():
    """Test analyze() handles API errors gracefully."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test-model",
    )
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        result = analyzer.analyze("Pod crash", "ns=test")
        assert result is not None
        assert "analysis" in result  # Should return fallback


def test_llm_analyze_timeout():
    """Test analyze() handles timeouts gracefully."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test-model",
        timeout=5,
    )
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = analyzer.analyze("Pod crash", "ns=test")
        assert result is not None
        assert "analysis" in result


def test_llm_build_prompt_no_prediction():
    """Test prompt building without prediction data."""
    analyzer = LLMAnalyzer(
        backend=LLMBackend.OLLAMA,
        url="http://localhost:11434",
        model="test",
    )
    prompt = analyzer.build_prompt(issue="Test issue", context="test context")
    assert "Test issue" in prompt
    assert "test context" in prompt
