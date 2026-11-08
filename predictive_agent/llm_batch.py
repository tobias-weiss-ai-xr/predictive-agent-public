"""LLM batch processing and optimization module for predictive-agent.

Provides:
- Request batching for similar analysis tasks
- LLM response caching
- Asynchronous processing
- Token estimation and optimization

This module extends the basic LLM functionality with performance optimizations.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from predictive_agent.l1_cache import get_cache
from predictive_agent.llm import LLMAnalyzer


@dataclass
class LLMCostEstimate:
    """Estimate of LLM call cost in tokens and time."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float = 0.0  # In USD
    estimated_time: float = 0.0  # In seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.estimated_cost,
            'estimated_time': self.estimated_time,
        }


@dataclass
class BatchRequest:
    """A single LLM request that can be batched with similar requests."""
    prompt: str
    context: Dict[str, Any]
    pod_key: str  # Identifier for the pod (e.g., "ns/name")
    priority: int = 0  # Higher priority requests are processed first
    created_at: float = field(default_factory=time.time)

    def get_key(self) -> str:
        """Generate a cache/deduplication key for this request."""
        # Simplify the prompt for grouping similar requests
        simplified = self._simplify_prompt(self.prompt)
        return hashlib.sha256(f"{simplified}".encode()).hexdigest()[:16]

    @staticmethod
    def _simplify_prompt(prompt: str) -> str:
        """Simplify a prompt to group similar requests together.
        
        This removes variable details (names, numbers, timestamps) while
        preserving the structure and intent.
        """
        # Replace pod names with placeholder
        import re
        simplified = re.sub(r'[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?', 'POD_NAME', prompt, flags=re.IGNORECASE)
        # Replace numeric values with placeholder
        simplified = re.sub(r'\d+\.?\d*', 'NUMBER', simplified)
        # Replace timestamps
        simplified = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 'TIMESTAMP', simplified)
        return simplified.lower().strip()


@dataclass
class BatchResult:
    """Result of a batched LLM request."""
    pod_key: str
    response: Dict[str, Any]
    cost_estimate: LLMCostEstimate
    processing_time: float
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pod_key': self.pod_key,
            'response': self.response,
            'cost_estimate': self.cost_estimate.to_dict(),
            'processing_time': self.processing_time,
            'cached': self.cached,
        }


class LLMBatchProcessor:
    """Process LLM requests in batches for improved efficiency.
    
    Features:
    - Groups similar prompts together
    - Caches common responses
    - Processes requests asynchronously
    - Tracks token usage and costs
    
    Usage:
        processor = LLMBatchProcessor(analyzer, max_batch_size=5, max_wait_time=2.0)
        
        # Queue requests
        for pod in pods_to_analyze:
            processor.queue_analysis(pod, context)
        
        # Process all queued requests
        results = await processor.process_all()
        
        # Or process with automatic flushing
        processor.start(auto_flush_interval=1.0)
        result = await processor.analyze_immediate(pod, context)
    """

    def __init__(
        self,
        analyzer: LLMAnalyzer,
        max_batch_size: int = 5,
        max_wait_time: float = 2.0,
        cache_enabled: bool = True,
        deduplication_enabled: bool = True,
        token_cost_per_1k: Optional[Dict[str, float]] = None,
    ):
        """Initialize the batch processor.
        
        Args:
            analyzer: LLMAnalyzer instance to use
            max_batch_size: Maximum number of requests to batch together
            max_wait_time: Maximum time to wait for batch to fill (seconds)
            cache_enabled: Whether to cache responses
            deduplication_enabled: Whether to deduplicate identical requests
            token_cost_per_1k: Token cost per 1000 tokens for cost estimation
                Format: {'input': cost_per_1k_input, 'output': cost_per_1k_output}
        """
        self._analyzer = analyzer
        self._max_batch_size = max_batch_size
        self._max_wait_time = max_wait_time
        self._cache_enabled = cache_enabled
        self._deduplication_enabled = deduplication_enabled
        self._token_cost = token_cost_per_1k or {
            'input': 0.00001,  # $0.01 per 1k input tokens (example)
            'output': 0.00002,  # $0.02 per 1k output tokens (example)
        }

        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._in_progress: Dict[str, asyncio.Future] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_tokens = 0
        self._total_cached = 0
        self._total_deduplicated = 0
        self._start_time = time.time()

        # Token estimation
        self._avg_output_tokens = 100.0  # Conservative estimate
        self._avg_input_tokens = 50.0

    def queue_analysis(
        self,
        prompt: str,
        context: Dict[str, Any],
        pod_key: str,
        priority: int = 0
    ) -> None:
        """Queue an analysis request for batch processing.
        
        Args:
            prompt: The LLM prompt
            context: Additional context for the request
            pod_key: Identifier for the pod (e.g., "ns/name")
            priority: Request priority (higher = processed first)
        """
        request = BatchRequest(
            prompt=prompt,
            context=context,
            pod_key=pod_key,
            priority=priority,
        )
        self._queue.put_nowait(request)
        self._total_requests += 1

    async def analyze_single(self, request: BatchRequest) -> BatchResult:
        """Analyze a single request with caching.
        
        Args:
            request: The batch request to process
            
        Returns:
            BatchResult with the analysis
        """
        cache = get_cache()

        # Check cache first (the cache hashes the raw prompt internally)
        if self._cache_enabled:
            hit, cached_response = cache.get_llm_prompt(request.prompt)
            if hit:
                self._total_cached += 1
                return BatchResult(
                    pod_key=request.pod_key,
                    response=cached_response,
                    cost_estimate=LLMCostEstimate(
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                    ),
                    processing_time=0.0,
                    cached=True,
                )

        # Deduplicate if enabled
        if self._deduplication_enabled:
            result = await cache.deduplicate_llm_call(
                request.prompt,
                lambda: self._execute_llm(request),
                estimated_tokens=self._estimate_tokens(request.prompt)
            )
        else:
            result = await self._execute_llm(request)

        # Cache the result
        if self._cache_enabled:
            cache.set_llm_prompt(request.prompt, result)

        # Estimate cost
        cost = self._estimate_cost(request.prompt, result)

        return BatchResult(
            pod_key=request.pod_key,
            response=result,
            cost_estimate=cost,
            processing_time=time.time() - request.created_at,
            cached=False,
        )

    async def _execute_llm(self, request: BatchRequest) -> Dict[str, Any]:
        """Execute the LLM call for a single request.
        
        Args:
            request: The batch request
            
        Returns:
            Parsed LLM response dict
        """
        start_time = time.time()
        try:
            result = self._analyzer.analyze(
                issue=request.context.get('issue', 'Unknown issue'),
                context=request.context.get('context', ''),
                prediction=request.context.get('prediction'),
            )
            return result
        finally:
            # Update token estimates based on actual usage
            _processing_time = time.time() - start_time
            # Very rough estimate: update averages based on processing time
            # This is a simplification - actual token counting would be better
            pass

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate the number of tokens in a prompt.
        
        This is a rough estimate using word count*.
        """
        # Very rough estimate: ~4 characters per token
        return len(prompt) // 4

    def _estimate_cost(self, prompt: str, response: Dict[str, Any]) -> LLMCostEstimate:
        """Estimate the cost of an LLM call.
        
        Args:
            prompt: The input prompt
            response: The LLM response
            
        Returns:
            LLMCostEstimate with estimated costs
        """
        input_tokens = self._estimate_tokens(prompt)

        # Estimate output tokens based on response size
        response_str = json.dumps(response)
        output_tokens = len(response_str) // 4

        total_tokens = input_tokens + output_tokens

        # Calculate cost
        input_cost = (input_tokens / 1000) * self._token_cost['input']
        output_cost = (output_tokens / 1000) * self._token_cost['output']
        estimated_cost = input_cost + output_cost

        # Estimate processing time (conservative)
        # Assuming ~50 tokens/second generation speed
        estimated_time = output_tokens / 50 if output_tokens > 0 else 0.1

        return LLMCostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
        )

    async def process_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Process a batch of requests.
        
        Args:
            requests: List of requests to process
            
        Returns:
            List of BatchResults in the same order as input
        """
        results = []

        # Sort by priority (highest first), then by creation time
        sorted_requests = sorted(
            requests,
            key=lambda r: (-r.priority, r.created_at)
        )

        # Process requests
        # For now, we process sequentially but with caching
        # Future improvement: parallel processing for independent requests
        for request in sorted_requests:
            result = await self.analyze_single(request)
            results.append(result)

            # Update average token counts
            self._update_token_estimates(result.cost_estimate)

        self._total_batches += 1
        return results

    def _update_token_estimates(self, cost_estimate: LLMCostEstimate) -> None:
        """Update token estimates based on actual usage."""
        # Exponential moving average
        alpha = 0.1  # Smoothing factor
        self._avg_input_tokens = (
            alpha * cost_estimate.input_tokens +
            (1 - alpha) * self._avg_input_tokens
        )
        self._avg_output_tokens = (
            alpha * cost_estimate.output_tokens +
            (1 - alpha) * self._avg_output_tokens
        )

    async def process_all(self) -> List[BatchResult]:
        """Process all queued requests until queue is empty.
        
        Returns:
            List of all BatchResults
        """
        all_results = []

        while not self._queue.empty():
            # Collect batch
            batch: list = []
            try:
                # Wait a bit for more requests to accumulate
                start_wait = time.time()
                while (
                    len(batch) < self._max_batch_size and
                    (time.time() - start_wait) < self._max_wait_time and
                    (not self._queue.empty() or len(batch) == 0)
                ):
                    if not self._queue.empty():
                        try:
                            request = self._queue.get_nowait()
                            batch.append(request)
                        except asyncio.QueueEmpty:
                            pass
                    else:
                        await asyncio.sleep(0.1)
            except TimeoutError:
                pass

            if batch:
                # Process the batch
                batch_results = await self.process_batch(batch)
                all_results.extend(batch_results)

        return all_results

    async def analyze_immediate(
        self,
        prompt: str,
        context: Dict[str, Any],
        pod_key: str,
        priority: int = 0
    ) -> BatchResult:
        """Analyze a request immediately without waiting for batch.
        
        Args:
            prompt: The LLM prompt
            context: Additional context
            pod_key: Pod identifier
            priority: Request priority
            
        Returns:
            BatchResult with the analysis
        """
        request = BatchRequest(
            prompt=prompt,
            context=context,
            pod_key=pod_key,
            priority=priority,
            created_at=time.time(),
        )
        return await self.analyze_single(request)

    def start(self, auto_flush_interval: float = 1.0) -> None:
        """Start the background processor.
        
        Args:
            auto_flush_interval: Time between automatic batch flushes (seconds)
        """
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(
            self._background_processor(auto_flush_interval)
        )

    async def _background_processor(self, interval: float) -> None:
        """Background task to process queued requests."""
        while self._running:
            try:
                # Wait for requests or timeout
                try:
                    # Wait with timeout
                    request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=interval
                    )
                    # Hand the woken-up request back so the batch drain
                    # below processes it (previously it was dropped here,
                    # silently losing one request per flush cycle).
                    self._queue.put_nowait(request)
                except TimeoutError:
                    # Timeout - process any pending requests
                    pass

                # Process any pending requests
                if not self._queue.empty():
                    await self.process_all()
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                # Log error but keep running
                import logging
                logger = logging.getLogger('LLMBatchProcessor')
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1.0)

    def stop(self) -> None:
        """Stop the background processor."""
        if not self._running:
            return

        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            self._processor_task = None

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        uptime = time.time() - self._start_time
        return {
            'total_requests': self._total_requests,
            'total_batches': self._total_batches,
            'total_cached': self._total_cached,
            'total_deduplicated': self._total_deduplicated,
            'uptime': uptime,
            'avg_input_tokens': self._avg_input_tokens,
            'avg_output_tokens': self._avg_output_tokens,
            'queue_size': self._queue.qsize() if hasattr(self._queue, 'qsize') else 0,
            'running': self._running,
        }


class ParallelLLMProcessor:
    """Process LLM requests in parallel for maximum throughput.
    
    Uses a semaphore to limit concurrency and avoid overloading the LLM server.
    """

    def __init__(
        self,
        analyzer: LLMAnalyzer,
        max_concurrent: int = 3,
        rate_limit: float = 10.0,  # requests per second
        cache_enabled: bool = True,
    ):
        """Initialize the parallel processor.
        
        Args:
            analyzer: LLMAnalyzer instance
            max_concurrent: Maximum concurrent requests
            rate_limit: Maximum requests per second (0 for unlimited)
            cache_enabled: Whether to enable caching
        """
        self._analyzer = analyzer
        self._max_concurrent = max_concurrent
        self._rate_limit = rate_limit
        self._cache_enabled = cache_enabled
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Rate limiting
        self._last_request_times: List[float] = []
        self._rate_lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_success = 0
        self._total_errors = 0

    async def analyze(
        self,
        prompt: str,
        context: Dict[str, Any],
        pod_key: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Analyze a single request with parallel processing.
        
        Args:
            prompt: The LLM prompt
            context: Additional context
            pod_key: Pod identifier (used for cache namespace)
            
        Returns:
            Tuple of (result, cached)
        """
        cache = get_cache()

        # Check cache - raw prompt (the cache hashes it internally)
        if self._cache_enabled:
            hit, cached_response = cache.get_llm_prompt(prompt)
            if hit:
                self._total_requests += 1
                self._total_success += 1
                return cached_response, True

        # Rate limiting
        await self._rate_limit_request()

        # Use semaphore for concurrency control
        async with self._semaphore:
            self._total_requests += 1
            try:
                # Execute the analysis
                result = await asyncio.to_thread(
                    self._analyzer.analyze,
                    issue=context.get('issue', 'Unknown issue'),
                    context=context.get('context', ''),
                    prediction=context.get('prediction'),
                )

                # Cache the result using raw prompt (shared across pods with same prompt)
                if self._cache_enabled:
                    cache.set_llm_prompt(prompt, result)

                self._total_success += 1
                return result, False
            except Exception as e:
                self._total_errors += 1
                return {
                    'analysis': f'Analysis failed: {str(e)}',
                    'severity': 'high',
                    'action': 'Retry analysis',
                    'command': '',
                }, False

    async def _rate_limit_request(self) -> None:
        """Apply rate limiting by tracking request times."""
        if self._rate_limit <= 0:
            return  # No rate limiting

        async with self._rate_lock:
            now = time.time()
            # Remove old requests (older than 1 second)
            self._last_request_times = [
                t for t in self._last_request_times
                if now - t < 1.0
            ]

            # Check if we can make another request
            if len(self._last_request_times) >= self._rate_limit:
                # Need to wait
                oldest = min(self._last_request_times)
                wait_time = 1.0 - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Update time after waiting
                    now = time.time()
                    self._last_request_times = [
                        t for t in self._last_request_times
                        if now - t < 1.0
                    ]

            # Record this request
            self._last_request_times.append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'total_requests': self._total_requests,
            'total_success': self._total_success,
            'total_errors': self._total_errors,
            'max_concurrent': self._max_concurrent,
            'rate_limit': self._rate_limit,
        }

    async def analyze_many(
        self,
        requests: List[Tuple[str, Dict[str, Any], str]],
    ) -> List[Tuple[Dict[str, Any], bool, str]]:
        """Analyze multiple requests in parallel.
        
        Args:
            requests: List of (prompt, context, pod_key) tuples
            
        Returns:
            List of (result, cached, pod_key) tuples
        """
        # Create tasks for all requests
        tasks = [
            self.analyze(prompt, context, pod_key)
            for prompt, context, pod_key in requests
        ]

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Combine with pod_keys
        return [
            (result[0], result[1], pod_key)
            for (result, pod_key) in zip(results, [r[2] for r in requests])
        ]
