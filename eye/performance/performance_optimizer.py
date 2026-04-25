"""
Performance Optimizer for ACES

Provides parallel processing, GPU acceleration, and performance monitoring
capabilities to optimize computational efficiency for military logistics simulations.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import psutil
import GPUtil
from datetime import datetime, timedelta
import threading
import queue
import json
import os

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    active_threads: int
    active_processes: int
    computation_time: float
    throughput: float  # operations per second


@dataclass
class ParallelTask:
    """Represents a task for parallel execution."""
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 1
    estimated_time: float = 1.0


@dataclass
class OptimizationResult:
    """Results from performance optimization."""
    task_id: str
    result: Any
    execution_time: float
    cpu_time: float
    memory_peak: float
    success: bool
    error_message: Optional[str] = None


class PerformanceOptimizer:
    """
    Performance optimization system for ACES simulations.

    Provides parallel processing, GPU acceleration, and comprehensive
    performance monitoring for large-scale military logistics scenarios.
    """

    def __init__(self, max_workers: Optional[int] = None, enable_gpu: bool = True):
        """
        Initialize the performance optimizer.

        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            enable_gpu: Whether to enable GPU acceleration
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_gpu = enable_gpu and (CUPY_AVAILABLE or TORCH_AVAILABLE)

        # Thread/process pools
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None

        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Task management
        self.task_queue = queue.PriorityQueue()
        self.results_queue = queue.Queue()
        self.active_tasks: Dict[str, ParallelTask] = {}

        # GPU resources
        self.gpu_available = self._check_gpu_availability()
        self.gpu_memory_pool = None

        # Initialize pools
        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize thread and process pools."""
        try:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        except Exception as e:
            print(f"Warning: Failed to initialize pools: {e}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU resources are available."""
        if not self.enable_gpu:
            return False

        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except:
            return False

    def start_monitoring(self, interval: float = 1.0):
        """
        Start performance monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        timestamp = datetime.now()

        # CPU and memory
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # GPU metrics
        gpu_usage = None
        gpu_memory = None
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUsed / gpus[0].memoryTotal * 100
            except:
                pass

        # Process information
        active_threads = threading.active_count()
        active_processes = len(mp.active_children())

        # Compute throughput (simplified)
        if self.metrics_history:
            time_diff = (timestamp - self.metrics_history[-1].timestamp).total_seconds()
            throughput = len(self.active_tasks) / max(time_diff, 0.1)
        else:
            throughput = 0.0

        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            active_threads=active_threads,
            active_processes=active_processes,
            computation_time=0.0,  # Updated by task execution
            throughput=throughput
        )

    def execute_parallel(
        self,
        tasks: List[ParallelTask],
        use_processes: bool = True,
        timeout: Optional[float] = None
    ) -> List[OptimizationResult]:
        """
        Execute tasks in parallel.

        Args:
            tasks: List of tasks to execute
            use_processes: Whether to use process pool (True) or thread pool (False)
            timeout: Maximum execution time per task

        Returns:
            List of execution results
        """
        if not tasks:
            return []

        start_time = time.time()
        results = []

        try:
            pool = self.process_pool if use_processes else self.thread_pool
            if not pool:
                # Fallback to sequential execution
                return self._execute_sequential(tasks)

            # Submit tasks
            future_to_task = {}
            for task in tasks:
                self.active_tasks[task.task_id] = task
                future = pool.submit(self._execute_task_wrapper, task)
                future_to_task[future] = task

            # Collect results
            for future in as_completed(future_to_task, timeout=timeout):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except Exception as e:
                    error_result = OptimizationResult(
                        task_id=task.task_id,
                        result=None,
                        execution_time=time.time() - start_time,
                        cpu_time=0.0,
                        memory_peak=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
                finally:
                    self.active_tasks.pop(task.task_id, None)

        except Exception as e:
            print(f"Parallel execution error: {e}")
            # Fallback to sequential
            return self._execute_sequential(tasks)

        return results

    def _execute_task_wrapper(self, task: ParallelTask) -> OptimizationResult:
        """Wrapper for task execution with performance monitoring."""
        start_time = time.time()
        start_cpu = time.process_time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = task.function(*task.args, **task.kwargs)

            execution_time = time.time() - start_time
            cpu_time = time.process_time() - start_cpu
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_peak = end_memory - start_memory

            return OptimizationResult(
                task_id=task.task_id,
                result=result,
                execution_time=execution_time,
                cpu_time=cpu_time,
                memory_peak=memory_peak,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return OptimizationResult(
                task_id=task.task_id,
                result=None,
                execution_time=execution_time,
                cpu_time=time.process_time() - start_cpu,
                memory_peak=0.0,
                success=False,
                error_message=str(e)
            )

    def _execute_sequential(self, tasks: List[ParallelTask]) -> List[OptimizationResult]:
        """Execute tasks sequentially as fallback."""
        results = []
        for task in tasks:
            result = self._execute_task_wrapper(task)
            results.append(result)
        return results

    def optimize_array_operations(
        self,
        arrays: List[np.ndarray],
        operation: str = 'sum',
        use_gpu: bool = True
    ) -> np.ndarray:
        """
        Optimize array operations using GPU acceleration if available.

        Args:
            arrays: List of numpy arrays
            operation: Operation to perform ('sum', 'mean', 'dot', etc.)
            use_gpu: Whether to use GPU acceleration

        Returns:
            Result of the operation
        """
        if not arrays:
            return np.array([])

        if use_gpu and self.enable_gpu and CUPY_AVAILABLE:
            try:
                # Convert to cupy arrays
                cp_arrays = [cp.asarray(arr) for arr in arrays]

                if operation == 'sum':
                    result = cp.sum(cp_arrays[0] if len(cp_arrays) == 1 else cp.stack(cp_arrays), axis=0)
                elif operation == 'mean':
                    result = cp.mean(cp_arrays[0] if len(cp_arrays) == 1 else cp.stack(cp_arrays), axis=0)
                elif operation == 'dot' and len(arrays) == 2:
                    result = cp.dot(cp_arrays[0], cp_arrays[1])
                else:
                    # Fallback to numpy
                    result = self._numpy_operation(arrays, operation)

                return cp.asnumpy(result)

            except Exception as e:
                print(f"GPU operation failed, falling back to CPU: {e}")

        # CPU fallback
        return self._numpy_operation(arrays, operation)

    def _numpy_operation(self, arrays: List[np.ndarray], operation: str) -> np.ndarray:
        """Perform numpy array operations."""
        if operation == 'sum':
            return np.sum(arrays[0] if len(arrays) == 1 else np.stack(arrays), axis=0)
        elif operation == 'mean':
            return np.mean(arrays[0] if len(arrays) == 1 else np.stack(arrays), axis=0)
        elif operation == 'dot' and len(arrays) == 2:
            return np.dot(arrays[0], arrays[1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def parallel_scenario_runs(
        self,
        scenario_function: Callable,
        scenario_configs: List[Dict[str, Any]],
        max_parallel: Optional[int] = None
    ) -> List[OptimizationResult]:
        """
        Run multiple scenarios in parallel.

        Args:
            scenario_function: Function that runs a single scenario
            scenario_configs: List of scenario configurations
            max_parallel: Maximum number of parallel runs

        Returns:
            List of scenario results
        """
        max_parallel = max_parallel or min(len(scenario_configs), self.max_workers)

        # Create tasks
        tasks = []
        for i, config in enumerate(scenario_configs):
            task = ParallelTask(
                task_id=f"scenario_{i}",
                function=scenario_function,
                args=(config,),
                kwargs={},
                priority=1,
                estimated_time=config.get('estimated_time', 10.0)
            )
            tasks.append(task)

        # Sort by priority and estimated time (shortest first for better load balancing)
        tasks.sort(key=lambda t: (t.priority, t.estimated_time))

        # Execute in batches to avoid overwhelming the system
        results = []
        batch_size = max_parallel

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = self.execute_parallel(batch, use_processes=True)
            results.extend(batch_results)

        return results

    def optimize_memory_usage(self, data_structures: List[Any]) -> Dict[str, Any]:
        """
        Optimize memory usage for large data structures.

        Args:
            data_structures: List of data structures to optimize

        Returns:
            Memory optimization report
        """
        report = {
            'original_memory': 0,
            'optimized_memory': 0,
            'compression_ratio': 1.0,
            'optimizations_applied': []
        }

        for ds in data_structures:
            if isinstance(ds, np.ndarray):
                original_size = ds.nbytes
                report['original_memory'] += original_size

                # Try different dtypes
                if ds.dtype == np.float64 and ds.max() < 1e6:
                    optimized = ds.astype(np.float32)
                    report['optimized_memory'] += optimized.nbytes
                    report['optimizations_applied'].append('dtype_reduction_float64_to_float32')
                elif ds.dtype == np.int64:
                    optimized = ds.astype(np.int32)
                    report['optimized_memory'] += optimized.nbytes
                    report['optimizations_applied'].append('dtype_reduction_int64_to_int32')
                else:
                    report['optimized_memory'] += original_size

            elif isinstance(ds, pd.DataFrame):
                original_size = ds.memory_usage(deep=True).sum()
                report['original_memory'] += original_size

                # DataFrame optimizations
                optimized = ds.copy()

                # Downcast numeric types
                for col in optimized.select_dtypes(include=[np.number]).columns:
                    if optimized[col].dtype == 'float64':
                        optimized[col] = pd.to_numeric(optimized[col], downcast='float')
                    elif optimized[col].dtype == 'int64':
                        optimized[col] = pd.to_numeric(optimized[col], downcast='integer')

                # Convert object columns to category if appropriate
                for col in optimized.select_dtypes(include=['object']).columns:
                    if optimized[col].nunique() / len(optimized) < 0.5:
                        optimized[col] = optimized[col].astype('category')

                optimized_size = optimized.memory_usage(deep=True).sum()
                report['optimized_memory'] += optimized_size

                if optimized_size < original_size:
                    report['optimizations_applied'].append('dataframe_optimization')

        if report['original_memory'] > 0:
            report['compression_ratio'] = report['optimized_memory'] / report['original_memory']

        return report

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}

        metrics_df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage,
                'gpu_usage': m.gpu_usage,
                'gpu_memory': m.gpu_memory,
                'active_threads': m.active_threads,
                'active_processes': m.active_processes,
                'throughput': m.throughput
            }
            for m in self.metrics_history
        ])

        report = {
            'summary': {
                'total_measurements': len(metrics_df),
                'monitoring_duration': (metrics_df['timestamp'].max() - metrics_df['timestamp'].min()).total_seconds(),
                'avg_cpu_usage': metrics_df['cpu_usage'].mean(),
                'max_cpu_usage': metrics_df['cpu_usage'].max(),
                'avg_memory_usage': metrics_df['memory_usage'].mean(),
                'max_memory_usage': metrics_df['memory_usage'].max(),
                'avg_throughput': metrics_df['throughput'].mean()
            },
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'gpu_available': self.gpu_available,
                'gpu_count': len(GPUtil.getGPUs()) if self.gpu_available else 0,
                'total_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            },
            'performance_trends': {
                'cpu_trend': self._calculate_trend(metrics_df['cpu_usage']),
                'memory_trend': self._calculate_trend(metrics_df['memory_usage']),
                'throughput_trend': self._calculate_trend(metrics_df['throughput'])
            }
        }

        if self.gpu_available and metrics_df['gpu_usage'].notna().any():
            report['summary'].update({
                'avg_gpu_usage': metrics_df['gpu_usage'].mean(),
                'max_gpu_usage': metrics_df['gpu_usage'].max(),
                'avg_gpu_memory': metrics_df['gpu_memory'].mean()
            })

        return report

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a metric series."""
        if len(series) < 2:
            return 'insufficient_data'

        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def export_performance_data(self, output_path: Path):
        """
        Export performance data to JSON.

        Args:
            output_path: Path to save the export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'performance_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_usage': m.cpu_usage,
                    'memory_usage': m.memory_usage,
                    'gpu_usage': m.gpu_usage,
                    'gpu_memory': m.gpu_memory,
                    'active_threads': m.active_threads,
                    'active_processes': m.active_processes,
                    'computation_time': m.computation_time,
                    'throughput': m.throughput
                }
                for m in self.metrics_history
            ],
            'system_capabilities': {
                'max_workers': self.max_workers,
                'gpu_available': self.gpu_available,
                'cupy_available': CUPY_AVAILABLE,
                'torch_available': TORCH_AVAILABLE
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()

        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        # Clear queues and tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except queue.Empty:
                break

        self.active_tasks.clear()
