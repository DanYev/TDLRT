#!/usr/bin/env python3
import sys
import inspect
import importlib

import workflow 


def get_module_functions(module):
    """Get all non-private functions from a module"""
    return {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith('_')}


def main():
    if len(sys.argv) < 2:
        print("Usage: exec.py <module> <command> [args...]")
        sys.exit(1)

    module_name = sys.argv[1]
    command = sys.argv[2]
    args = sys.argv[3:]

    modules = {
        'workflow': workflow,
    }

    if module_name not in modules:
        raise ValueError(f"Unknown module: {module_name}. Available modules: {', '.join(modules.keys())}")

    module = modules[module_name]
    functions = get_module_functions(module)

    if command not in functions:
        raise ValueError(f"Unknown command: {command}. Available commands for {module_name}: {', '.join(functions.keys())}")

    try:
        functions[command](*args)
    except Exception as e:
        print(f"Error executing {module_name}.{command}: {str(e)}")
        raise


if __name__ == "__main__":
    main()
