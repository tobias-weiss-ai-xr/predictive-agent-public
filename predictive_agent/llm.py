"""Multi-backend LLM analyzer (Ollama, SAIA, TUD, OpenAI)."""

import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """LLM backend enum."""
    OLLAMA = "ollama"
    SAIA = "saia"
    TUD = "tud"
    OPENAI = "openai"
    JEVLOCAL = "jevlocal"


class LLMAnalyzer:
    """Multi-backend LLM analyzer for Kubernetes health analysis."""

    def __init__(self, backend, url, model, api_key=None, timeout=300, max_tokens=4096):
        """Initialize LLM analyzer.
        
        Args:
            backend: LLMBackend enum value
            url: API endpoint URL
            model: Model name
            api_key: Optional API key (required for SAIA, TUD, OpenAI)
            timeout: API request timeout in seconds
            max_tokens: Max output tokens for OpenAI-compatible calls. Must be
                large enough for reasoning + final JSON: with max_tokens=1024
                DeepSeek V4 Flash burned the whole budget on reasoning_content
                and returned content=None, silently degrading every analysis.
        """
        self.backend = backend
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def build_prompt(self, issue, context, prediction=None):
        """Build LLM prompt with issue, context, and prediction data.
        
        Args:
            issue: Issue description
            context: Context information
            prediction: Optional prediction data dict with keys:
                - risk_score: Risk score (0-1)
                - ttf_minutes: Time to failure in minutes
                - confidence: Confidence score (0-1)
                - markov_state: Markov state string
                - memory_trend: Memory trend value
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        prompt_parts.append(f"Issue: {issue}")
        prompt_parts.append(f"Context: {context}")

        if prediction:
            prompt_parts.append("\nPrediction data:")
            if "risk_score" in prediction:
                prompt_parts.append(f"  Risk score: {prediction['risk_score']}")
            if "ttf_minutes" in prediction:
                prompt_parts.append(f"  Time to failure: {prediction['ttf_minutes']} minutes")
            if "confidence" in prediction:
                prompt_parts.append(f"  Confidence: {prediction['confidence']}")
            if "markov_state" in prediction:
                prompt_parts.append(f"  Markov state: {prediction['markov_state']}")
            if "memory_trend" in prediction:
                prompt_parts.append(f"  Memory trend: {prediction['memory_trend']}")

        prompt_parts.append("\nAnalyze the issue and provide:")
        prompt_parts.append("- analysis: Brief analysis of the root cause")
        prompt_parts.append("- severity: high, medium, or low")
        prompt_parts.append("- action: Recommended action")
        prompt_parts.append("- command: kubectl command to execute (if applicable)")
        prompt_parts.append(
            "Respond with ONLY a JSON object containing exactly these keys: "
            "analysis, severity, action, command. No markdown, no prose "
            "outside the JSON."
        )

        return "\n".join(prompt_parts)

    def analyze(self, issue, context, prediction=None):
        """Perform analysis using the configured LLM backend.
        
        Args:
            issue: Issue description
            context: Context information
            prediction: Optional prediction data dict
            
        Returns:
            Parsed analysis result dict
        """
        prompt = self.build_prompt(issue, context, prediction)

        try:
            import urllib.error
            import urllib.request

            if self.backend == LLMBackend.JEVLOCAL:
                # Local Jev-style decision model: calibrated choice probabilities
                # from the laptop-GPU decide endpoint, no free text. Map the
                # argmax back to the severity/action schema every caller expects.
                question, options = self.build_decision_prompt(issue, context, prediction)
                jdata = json.dumps({"question": question, "options": options}).encode("utf-8")
                jreq = urllib.request.Request(
                    f"{self.url.rstrip('/')}/decide",
                    data=jdata,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(jreq, timeout=self.timeout) as response:
                    res = json.loads(response.read().decode("utf-8"))
                if not res.get("option_ids") or not res.get("probabilities"):
                    raise ValueError("decision endpoint returned no probabilities")
                return self._jev_result(res)

            if self.backend == LLMBackend.OLLAMA:
                endpoint = f"{self.url.rstrip('/')}/api/chat"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
                headers = {"Content-Type": "application/json"}
            else:
                # OpenAI compatible (SAIA, TUD, OPENAI)
                endpoint = f"{self.url.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                }
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(endpoint, data=data, headers=headers)

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                res_json = json.loads(body)

                if self.backend == LLMBackend.OLLAMA:
                    # Ollama returns content in 'message' -> 'content' or similar depending on version,
                    # but tests expect 'response' field
                    content = res_json.get("response", res_json.get("message", {}).get("content", ""))
                else:
                    # OpenAI compatible: choices[0].message.content
                    choices = res_json.get("choices", [])
                    content = choices[0].get("message", {}).get("content", "") if choices else ""

                return self.parse_response(content)

        except urllib.error.HTTPError as e:
            # HTTP rejection (400/401/403/...) is a request/config problem,
            # not a connectivity loss — say so instead of blaming the network.
            message = f"LLM API rejected request (HTTP {e.code}): {e.reason}"
            logger.warning(message)
            return {
                "analysis": message,
                "severity": "medium",
                "action": f"Fix LLM request (HTTP {e.code})",
                "command": "",
            }
        except TimeoutError:
            # A slow/loaded LLM is not a connectivity fault; mislabeling it as
            # "check connectivity" generated weeks of identical false analyses.
            message = f"LLM request timed out after {self.timeout}s"
            logger.warning(message)
            return {
                "analysis": message,
                "severity": "medium",
                "action": "Retry, LLM timed out",
                "command": "",
            }
        except (urllib.error.URLError, OSError) as e:
            # Connectivity loss — the network path to the LLM is broken, so
            # analysis could not be produced.
            # TimeoutError is an OSError subclass, so it is caught by its own
            # clause above and never misclassified as connectivity loss.
            message = f"LLM API error: {str(e)}"
            logger.warning(message)
            return {
                "analysis": message,
                "severity": "medium",
                "action": "Check LLM connectivity",
                "command": "",
            }
        except Exception as e:
            message = f"LLM analysis error: {str(e)}"
            logger.warning(message)
            return {
                "analysis": message,
                "severity": "medium",
                "action": "Monitor, analysis deferred",
                "command": "",
            }

    def parse_response(self, response):
        """Parse LLM JSON response with fallback for invalid JSON.
        
        Tolerates reasoning-model output: markdown code fences around the JSON
        and surrounding prose are stripped; the first balanced JSON object in
        the response is used.
        """
        try:
            parsed = json.loads(self._extract_json(response))
            # Ensure required fields exist
            if not isinstance(parsed, dict):
                return self._fallback_response()

            # Add missing required fields with defaults
            if "analysis" not in parsed:
                parsed["analysis"] = "Unknown issue"
            if "severity" not in parsed:
                parsed["severity"] = "medium"
            if "action" not in parsed:
                parsed["action"] = "Monitor"
            if "command" not in parsed:
                parsed["command"] = ""

            return parsed
        except (json.JSONDecodeError, TypeError, AttributeError):
            return self._fallback_response()

    @staticmethod
    def _extract_json(response):
        """Return the first balanced JSON object in a model response."""
        if not isinstance(response, str):
            return response
        text = response.strip()
        # Strip markdown code fences (e.g. ```json ... ``` around the JSON).
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) >= 3 else text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return text[start:]

    def build_decision_prompt(self, issue, context, prediction=None):
        """Build the (question, options) pair for a local Jev-style decision.

        The decision model scores three risk grades and returns calibrated
        probabilities; it never generates free text (one forward pass)."""
        question = (
            f"An operator agent monitors a Kubernetes workload.\n"
            f"Issue: {issue}\nContext: {context}"
        )
        if prediction:
            question += (
                f"\nPrediction: risk_score={prediction.get('risk_score')}, "
                f"ttf_minutes={prediction.get('ttf_minutes')}, "
                f"confidence={prediction.get('confidence')}, "
                f"markov_state={prediction.get('markov_state')}"
            )
        question += (
            "\nGiven all this, how likely is the workload to keep failing "
            "without intervention?"
        )
        options = [
            {"id": "low", "text": "Low risk - it will stabilise or self-heal, no action needed"},
            {"id": "moderate", "text": "Moderate - monitor closely and schedule a review"},
            {"id": "high", "text": "High risk - it will keep failing, needs active remediation"},
        ]
        return question, options

    @staticmethod
    def _jev_result(res):
        """Map a decide-endpoint response onto the analysis schema."""
        ids = res.get("option_ids", [])
        probs = res.get("probabilities", [])
        choice = res.get("choice", 0)
        calibrated = bool(res.get("calibrated"))
        choice_id = ids[choice] if ids and 0 <= choice < len(ids) else (ids[0] if ids else "low")
        severity = {"low": "low", "moderate": "medium", "high": "high"}.get(choice_id, "medium")
        action = {
            "low": "Monitor",
            "moderate": "Escalate for review",
            "high": "Remediate",
        }.get(choice_id, "Monitor")
        labels = {"low": "low", "moderate": "moderate", "high": "high"}
        odds = ", ".join(f"{labels.get(i, i)}={p:.2f}" for i, p in zip(ids, probs))
        return {
            "analysis": f"Jev decision: P({odds}); chose {choice_id} (calibrated={calibrated})",
            "severity": severity,
            "action": action,
            "command": "",
            "jev_choice": choice_id,
            "jev_option_ids": ids,
            "jev_probabilities": probs,
            "jev_calibrated": calibrated,
            "jev_confidence": round(max(probs), 4) if probs else 0.0,
        }

    def _fallback_response(self):
        """Generate fallback response for invalid JSON."""
        logger.warning(
            "LLM analysis degraded: response was not valid JSON (Monitor fallback)"
        )
        return {
            "analysis": "Unknown issue - invalid LLM response",
            "severity": "medium",
            "action": "Monitor",
            "command": ""
        }
