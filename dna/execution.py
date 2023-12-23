from __future__ import annotations

from typing import Optional, Any
from abc import ABCMeta, abstractmethod
import time
from datetime import timedelta
from enum import Enum
from logging import Logger


class ExecutionState(Enum):
    NOT_STARTED = 0
    '''Execution is not started.'''
    STARTING = 1
    '''Execution is preparing for start.'''
    RUNNING = 2
    '''Execution is running.'''
    STOPPING = 3
    '''Execution is stopping.'''
    STOPPED = 4
    '''Execution has been stopped by something.'''
    FAILED = 5
    '''Execution has been finished because of a failure'''
    COMPLETED = 6
    '''Execution has been done sucessfully.'''


class Execution(metaclass=ABCMeta):
    @abstractmethod
    def run(self) -> Any:
        """본 실행을 수행시킨다.

        Returns:
            Any: 실행 완료로 생성된 결과.
        """
        pass


class ExecutionContext(metaclass=ABCMeta):
    @abstractmethod
    def started(self) -> None:
        """실행이 시작됨을 알린다.
        """
        pass
    
    @abstractmethod
    def report_progress(self, progress:Any) -> None:
        """실행 진행 상황을 알린다.

        Args:
            progress (Any): 진행 상황 정보.
        """
        pass

    @abstractmethod
    def completed(self, result:Any) -> None:
        """실행이 완료됨을 알린다.
        '실행 완료'는 실행이 그 목적을 모두 달성하고 종료되는 것을 믜미한다.

        Args:
            result (Any): 실행 완료로 생성된 결과 객체.
        """
        pass

    @abstractmethod
    def stopped(self, details:str) -> None:
        """사용자에 의해 실행이 중단됨을 알린다.

        Args:
            details (str): 실행 중단 원인.
        """
        pass

    @abstractmethod
    def failed(self, cause:str) -> None:
        """실행 도중 오류가 발생하여 종료됨을 알린다.

        Args:
            cause (str): 오류 원인.
        """
        pass


class NoOpExecutionContext(ExecutionContext):
    def started(self) -> None: pass
    def report_progress(self, progress:Any) -> None: pass
    def completed(self, result:Any) -> None: pass
    def stopped(self, details:str) -> None: pass
    def failed(self, cause:str) -> None: pass


class LoggingExecutionContext(ExecutionContext):
    def __init__(self, logger:Optional[Logger]=None) -> None:
        super().__init__()
        self.logger = logger

    def started(self) -> None:
        if self.logger is not None:
            self.logger.info(f'started')

    def report_progress(self, progress:Any) -> None: 
        if self.logger is not None:
            self.logger.info(f'progress reported: progress={progress}')

    def completed(self, result:Any) -> None:
        if self.logger is not None:
            self.logger.info(f'completed: result={result}')

    def stopped(self, details:str) -> None:
        if self.logger is not None:
            self.logger.info(f'stopped: details={details}')

    def failed(self, cause:str) -> None:
        if self.logger is not None:
            self.logger.info(f'failed: cause={cause}')

class ExecutionFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, context:ExecutionContext) -> Execution:
        """새로운 실행 객체를 생성한다.
        본 메소드를 통해 실행 객체가 생성만되지, 그 실행이 시작되지는 않는다.

        Args:
            context (ExecutionContext): 실행 과정에서 사용될 문맥 정보 객체.

        Returns:
            Execution: 생성된 실행 객체.
        """
        pass



class CancellationError(Exception):
    def __init__(self, message:Optional[str]) -> None:
        self.message = message
        super().__init__(message)


import threading
class AbstractExecution(Execution):
    def __init__(self, context:Optional[ExecutionContext]=None):
        self._ctx = context if context else NoOpExecutionContext()
        
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.state = ExecutionState.NOT_STARTED
        self.stop_details = None
        
    @property
    def context(self) -> ExecutionContext:
        return self._ctx
        
    @abstractmethod
    def run_work(self) -> Any:
        """본 실행이 수행해야 하는 작업을 수행한다.
        본 메소드는 ``run()`` 메소드가 호출되는 경우, 자동적으로 수행된다.

        Returns:
            Any: 실행 결과.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """본 실행이 종료될 때, 종료화 작업을 수행한다.
        """
        pass
    
    def check_stopped(self) -> None:
        with self.lock:
            if self.state == ExecutionState.STOPPING:
                raise CancellationError(self.stop_details)

    def run(self) -> Any:
        with self.lock:
            if self.state != ExecutionState.NOT_STARTED:
                raise AssertionError(f'invalid execution state: {self.state.name}, ',
                                     f'expected={ExecutionState.NOT_STARTED.name}')
            self.state = ExecutionState.RUNNING
            self.cond.notify_all()
        self._ctx.started()
        
        try:
            result = self.run_work()
            
            state = None
            with self.lock:
                if self.state == ExecutionState.RUNNING:
                    state = self.state = ExecutionState.COMPLETED
                elif self.state == ExecutionState.STOPPING:
                    state = self.state = ExecutionState.STOPPED
                else:
                    raise AssertionError(f'invalid execution state: {self.state.name}, '
                                            f'expected={ExecutionState.RUNNING.name}')
                self.cond.notify_all()
            if state == ExecutionState.COMPLETED:
                self._ctx.completed(result)
            elif state == ExecutionState.STOPPED:
                self._ctx.stopped('user requested')
            return result
        except CancellationError as e:
            with self.lock:
                if self.state != ExecutionState.RUNNING and self.state != ExecutionState.STOPPING:
                    raise AssertionError(f'invalid execution state: {self.state.name}, expected={ExecutionState.RUNNING.name}')
                self.state = ExecutionState.STOPPED
                self.cond.notify_all()
            self._ctx.stopped(str(e))
        except Exception as e:
            with self.lock:
                self.state = ExecutionState.FAILED
                self.cond.notify_all()
            self._ctx.failed(str(e))
            raise e
        finally:
            self.finalize()
        
    def stop(self, details: str='user requested', nowait=False) -> None:
        with self.lock:
            if self.state == ExecutionState.RUNNING:
                self.stop_details = details
                self.state = ExecutionState.STOPPING
            else:
                return
                
        self.stop_work()        
        if not nowait:
            with self.lock:
                while self.state == ExecutionState.STOPPING:
                    self.cond.wait()
                    
    @abstractmethod
    def stop_work(self) -> None: pass


class InvocationError(Exception):
    def __init__(self, message:Optional[str]) -> None:
        self.message = message
        super().__init__(message)


class TimeoutError(Exception):
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(message)


class AsyncExecution(Execution):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.state = ExecutionState.NOT_STARTED
        self.result = None
        self.message:Optional[str] = None

    def is_started(self) -> bool:
        with self.lock: return self.state.value >= ExecutionState.RUNNING.value

    def is_running(self) -> bool:
        with self.lock: return self.state == ExecutionState.RUNNING

    def is_completed(self) -> bool:
        with self.lock: return self.state == ExecutionState.COMPLETED

    def is_failed(self) -> bool:
        with self.lock: return self.state == ExecutionState.FAILED

    def is_stopped(self) -> bool:
        with self.lock: return self.state == ExecutionState.STOPPED

    def wait_for_finished(self, timeout:float=timedelta.max.total_seconds()) -> ExecutionState:
        due = time.time() + timeout
        with self.lock:
            while True:
                timeout = due - time.time()
                if self.cond.wait_for(lambda: self.state.value >= ExecutionState.STOPPED.value, timeout):
                    return self.state
                else:
                    raise TimeoutError(f"timeout={timedelta(seconds=timeout)}")

    def get(self, timeout:float=timedelta.max.total_seconds()) -> Any:
        due = time.time() + timeout
        with self.lock:
            while True:
                timeout = due - time.time()
                if self.cond.wait_for(lambda: self.state.value >= ExecutionState.STOPPED.value, timeout):
                    if self.state == ExecutionState.COMPLETED:
                        return self.result
                    elif self.state == ExecutionState.STOPPED:
                        raise CancellationError(self.message)
                    elif self.state == ExecutionState.FAILED:
                        raise InvocationError(self.message)
                    else:
                        raise AssertionError(f"unexpected state: {self.state}")
                else:
                    raise TimeoutError(f"timeout={timedelta(seconds=timeout)}")

    def notify_started(self) -> None:
        with self.lock:
            self.state = ExecutionState.RUNNING
            self.cond.notify_all()

    def notify_completed(self, result: Any) -> None:
        with self.lock:
            self.result = result
            self.state = ExecutionState.COMPLETED
            self.cond.notify_all()

    def notify_stopped(self, message: str) -> None:
        with self.lock:
            self.message = message
            self.state = ExecutionState.STOPPED
            self.cond.notify_all()

    def notify_failed(self, message: str) -> None:
        with self.lock:
            self.message = message
            self.state = ExecutionState.FAILED
            self.cond.notify_all()