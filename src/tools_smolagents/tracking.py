from functools import wraps
from typing import Callable, List, Dict, Any
import inspect
from datetime import datetime
import functools
from copy import deepcopy

def create_function_tracker(tracking_list: List[Dict[str, Any]]):
    """
    Creates a function wrapper that tracks usage in the provided tracking list.
    """
    def track_function_usage(name: str, func: Callable) -> Callable:
        """
        Wraps a function to track its usage.
        """
        unbound_func = getattr(func, '__func__', func)
        
        @wraps(unbound_func)
        def wrapper(*args, **kwargs):
            # The signature requires the instance ('self') for correct binding.
            # The functools.partial call ensures `args[0]` is that instance.
            sig = inspect.signature(unbound_func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            non_default_params = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name in ('self', 'cls'):
                    continue
                param = sig.parameters[param_name]
                if param.default is inspect.Parameter.empty or param_value != param.default:
                    non_default_params[param_name] = param_value
            
            output = None
            exception = None
            try:
                # ✨ FINAL FIX: We know the original `func` is a special callable that
                # doesn't want the `self` argument passed to it again.
                # We slice off the instance (`args[0]`) and pass the rest of the args.
                output = func(*args[1:], **kwargs)

            except Exception as e:
                exception = e
            
            usage_record = {
                "timestamp": datetime.now().isoformat(),
                "function_name": name,
                "actual_function": unbound_func.__name__,
                "parameters": non_default_params,
                "output": output,
                "error": str(exception) if exception else None
            }
            
            tracking_list.append(usage_record)
            
            if exception:
                raise exception
                
            return output
        
        return wrapper
    
    return track_function_usage

def wrap_tool(tool, wrapper, prefix=""):
    original_method = getattr(tool, 'forward')
    wrapped_func = wrapper(prefix+tool.name, original_method)
    bound_wrapper = functools.partial(wrapped_func, tool)
    setattr(tool, 'forward', bound_wrapper)
    return tool