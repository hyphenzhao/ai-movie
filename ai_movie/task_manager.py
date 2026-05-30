"""Thread-safe task registry for tracking background work."""

import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Task:
    name: str
    thread: threading.Thread | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class TaskManager:
    """Tracks running background tasks. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}

    def start(self, name: str, target: Callable, args: tuple = ()) -> Task:
        """Start a new background task."""
        task = Task(name=name)
        task.thread = threading.Thread(
            target=target,
            args=args,
            daemon=True,
            name=name,
        )
        with self._lock:
            self._tasks[name] = task
        task.thread.start()
        return task

    def cancel(self, name: str):
        """Signal a task to cancel."""
        with self._lock:
            task = self._tasks.get(name)
        if task:
            task.cancel_event.set()

    def finish(self, name: str):
        """Mark a task as finished (called by the task itself on completion)."""
        with self._lock:
            self._tasks.pop(name, None)

    def get_active_names(self) -> list[str]:
        """Return names of tasks that are still running."""
        with self._lock:
            return [
                name
                for name, task in self._tasks.items()
                if task.thread and task.thread.is_alive()
            ]

    @property
    def busy(self) -> bool:
        return len(self.get_active_names()) > 0

    def shutdown(self, timeout: float = 3.0) -> list[str]:
        """Cancel all tasks and wait for them to finish. Returns names of tasks that didn't stop in time."""
        active = self.get_active_names()
        for name in active:
            self.cancel(name)
        still_running = []
        for name in active:
            with self._lock:
                task = self._tasks.get(name)
            if task and task.thread and task.thread.is_alive():
                task.thread.join(timeout=timeout)
                if task.thread.is_alive():
                    still_running.append(name)
        return still_running


# Global singleton
task_manager = TaskManager()
