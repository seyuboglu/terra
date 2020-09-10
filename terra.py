import luigi

def process(fn):
    """
    Decorator for wrapping functions with
    """

    @wraps(fn)
    def with_logging(*args, **kwargs):
        is_logged = kwargs.pop("is_logged", True)
        args_dict = getcallargs(fn, *args, **kwargs)

        process_dir = args_dict.get("process_dir", None)
        if process_dir is None:
            process_dir = _get_process_dir(fn)

        if is_logged:
            process_dir = args_dict.get("process_dir", None)
            if process_dir is None:
                process_dir = _get_process_dir(fn)
            run_dir = _get_next_run_dir(process_dir)

            params_dict = {
                "git": log_git_status(run_dir),
                "notebook": "get_ipython" in globals().keys(),
                "start_time": datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                "module": fn.__module__,
                "fn": fn.__name__,
                "kwargs": args_dict,
            }

            write_input(run_dir, params_dict)

            logger = init_logging(os.path.join(run_dir, "process.log"))
            args_dict["process_dir"] = run_dir
            print(f"process: running in directory {run_dir}")

            # load node inputs
            for key, value in args_dict.items():
                if isinstance(value, ProcessOutput):
                    str_rep = json.dumps(value.serialize(), indent=2)
                    print(f"Loading process output: {str_rep} \n and passing to parameter '{key}'")
                    args_dict[key] = value.load()
                if isinstance(value, ProcessInput):
                    str_rep = json.dumps(value.serialize(), indent=2)
                    print(f"Loading process output: {str_rep} \n and passing to parameter '{key}'")
                    args_dict[key] = value.load()

            try:
                out = fn(**args_dict)
            except (Exception, KeyboardInterrupt) as e:
                time = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
                except_dir = os.path.join(process_dir, "_runs", "_except", time)
                # TODO: decide what to do with exceptions
                # shutil.move(run_dir, except_dir)
                print(traceback.format_exc())
                raise e
            else:
                if out is not None:
                    _write_output(out, run_dir)
            del logger

            return out
        else:
            return fn(**args_dict)

    return with_logging 