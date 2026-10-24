# 第7步A100 pilot：测试门槛停止记录

执行日期：**2026-10-25（Asia/Singapore）**，文件名日期沿用任务约定。用户已批准[训练计划](terrain_controller_pilot_plan_20261024.md)，执行源码为`9785c012cfd51b113fd8dc245034a6e46f98d36f`。

**最终测试结果为81通过、1失败、0跳过。按本轮“任何测试失败即停止并报告traceback”的条件，已停止后续操作，pilot未启动。** 失败来自pandas缺少依赖pytz；训练实现和超参数未修改。

## 1. 已完成的准备

- A100服务器新建独立任务目录，仓库检出批准版本。使用Git稀疏检出取得源码、测试、配置、文档和工具，未复制历史图片。
- 检查全部8张A100的UUID、显存、利用率与计算PID，选出当时空闲的物理GPU1、GPU3作为候选，每卡80GiB。本轮没有启动GPU进程；这些卡没有被占用或保留，后续启动时需重新检查。
- 环境为Python3.11.17、Torch2.5.1+cu124、NumPy1.26.4、pytest9.1.1，Kimodo来自已有项目资源。测试使用`CUDA_VISIBLE_DEVICES=''`，OMP/OpenBLAS/MKL线程数为1，`TMPDIR`设为项目临时目录。
- 先执行不依赖真实数据的单元测试。源数据复制、逐文件SHA-256核对和GPU冒烟尚未开始。

## 2. 环境准备与失败

第一次测试得到`81 passed, 1 skipped, 11 warnings in 9.93s`。跳过的`test_pickle_dict_of_rows_is_keyed_by_filename`缺少pandas。

随后在本任务的独立目录补装pandas2.3.3，命令为：

```bash
$PY -m pip install --target "$STEP7_ROOT/runtime_extra" --no-deps pandas==2.3.3
export PYTHONPATH="$STEP7_ROOT/runtime_extra:$KIMODO_ROOT"
```

这条补装命令使用了`--no-deps`，没有同时安装pytz。基础环境保持原样，但新的pandas仍不能导入。环境准备未完成时复跑测试，得到：

```text
1 failed, 81 passed, 11 warnings in 7.80s
```

失败测试：`tests/test_bones_terrain_survey.py::test_pickle_dict_of_rows_is_keyed_by_filename`。

```text
ImportError: Unable to import required dependencies:
pytz: No module named 'pytz'
```

实际测试命令如下，退出码为1：

```bash
$PY -m pytest tests/test_bones_env_dataset.py tests/test_terrain_controller.py \
  tests/test_skeleton_ray_sampler.py tests/test_bones_terrain_fit.py \
  tests/test_bones_terrain_scene.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py -q --full-trace --color=no \
  --junitxml=runs/tmt_pilot_coord/tests_final.xml
```

<details>
<summary>完整失败traceback（仅将私有目录替换为角色变量）</summary>

```text
cls = <class '_pytest.runner.CallInfo'>
func = <function call_and_report.<locals>.<lambda> at 0x7fd8c85839c0>
when = 'call'
reraise = (<class '_pytest.outcomes.Exit'>, <class 'KeyboardInterrupt'>)

    @classmethod
    def from_call(
        cls,
        func: Callable[[], TResult],
        when: Literal["collect", "setup", "call", "teardown"],
        reraise: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> CallInfo[TResult]:
        """Call func, wrapping the result in a CallInfo.
    
        :param func:
            The function to call. Called without arguments.
        :type func: Callable[[], _pytest.runner.TResult]
        :param when:
            The phase in which the function is called.
        :param reraise:
            Exception or exceptions that shall propagate if raised by the
            function, instead of being wrapped in the CallInfo.
        """
        excinfo = None
        instant = timing.Instant()
        try:
>           result: TResult | None = func()
                                     ^^^^^^

../../env/lib/python3.11/site-packages/_pytest/runner.py:361: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>       lambda: runtest_hook(item=item, **kwds),
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        when=when,
        reraise=get_reraise_exceptions(item.config),
    )

../../env/lib/python3.11/site-packages/_pytest/runner.py:250: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <HookCaller 'pytest_runtest_call'>
kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}

    def __call__(self, **kwargs: object) -> Any:
        """Call the hook.
    
        Only accepts keyword arguments, which should match the hook
        specification.
    
        Returns the result(s) of calling all registered plugins, see
        :ref:`calling`.
        """
        assert not self.is_historic(), (
            "Cannot directly call a historic hook - use call_historic instead."
        )
        self._verify_all_args_are_provided(kwargs)
        firstresult = self.spec.opts.get("firstresult", False) if self.spec else False
        # Copy because plugins may register other plugins during iteration (#438).
>       return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_hooks.py:512: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.config.PytestPluginManager object at 0x7fd9b2b4b490>
hook_name = 'pytest_runtest_call'
methods = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _hookexec(
        self,
        hook_name: str,
        methods: Sequence[HookImpl],
        kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        # called from all hookcaller instances.
        # enable_tracing will set its own wrapping function at self._inner_hookexec
>       return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_manager.py:120: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
caller_kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
                        res = hook_impl.function(*args)
                        if res is not None:
                            results.append(res)
                            if firstresult:  # halt further impl calls
                                break
            except BaseException as exc:
                exception = exc
        finally:
            if firstresult:  # first result hooks return a single value
                result = results[0] if results else None
            else:
                result = results
    
            # run all wrapper post-yield blocks
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        try:
                            teardown.throw(exception)
                        except RuntimeError as re:
                            # StopIteration from generator causes RuntimeError
                            # even for coroutine usage - see #544
                            if (
                                isinstance(exception, StopIteration)
                                and re.__cause__ is exception
                            ):
                                teardown.close()
                                continue
                            else:
                                raise
                    else:
                        teardown.send(result)
                    # Following is unreachable for a well behaved hook wrapper.
                    # Try to force finalizers otherwise postponed till GC action.
                    # Note: close() may raise if generator handles GeneratorExit.
                    teardown.close()
                except StopIteration as si:
                    result = si.value
                    exception = None
                    continue
                except BaseException as e:
                    exception = e
                    continue
                _raise_wrapfail(teardown, "has second yield")
    
        if exception is not None:
>           raise exception

../../env/lib/python3.11/site-packages/pluggy/_callers.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
caller_kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
                        res = hook_impl.function(*args)
                        if res is not None:
                            results.append(res)
                            if firstresult:  # halt further impl calls
                                break
            except BaseException as exc:
                exception = exc
        finally:
            if firstresult:  # first result hooks return a single value
                result = results[0] if results else None
            else:
                result = results
    
            # run all wrapper post-yield blocks
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        try:
>                           teardown.throw(exception)

../../env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>
item = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: nodes.Item) -> Generator[None]:
        self.log_cli_handler.set_when("call")
    
        with self._runtest_for(item, "call"):
>           yield

../../env/lib/python3.11/site-packages/_pytest/logging.py:865: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
caller_kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
                        res = hook_impl.function(*args)
                        if res is not None:
                            results.append(res)
                            if firstresult:  # halt further impl calls
                                break
            except BaseException as exc:
                exception = exc
        finally:
            if firstresult:  # first result hooks return a single value
                result = results[0] if results else None
            else:
                result = results
    
            # run all wrapper post-yield blocks
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        try:
>                           teardown.throw(exception)

../../env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <CaptureManager _method='fd' _global_capturing=<MultiCapture out=<FDCapture 1 oldfd=5 _state='suspended' tmpfile=<_io....xtIOWrapper name='/dev/null' mode='r' encoding='utf-8'>> _state='suspended' _in_suspended=False> _capture_fixture=None>
item = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None]:
        with self.item_capture("call", item):
>           return (yield)
                    ^^^^^

../../env/lib/python3.11/site-packages/_pytest/capture.py:900: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
caller_kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
                        res = hook_impl.function(*args)
                        if res is not None:
                            results.append(res)
                            if firstresult:  # halt further impl calls
                                break
            except BaseException as exc:
                exception = exc
        finally:
            if firstresult:  # first result hooks return a single value
                result = results[0] if results else None
            else:
                result = results
    
            # run all wrapper post-yield blocks
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        try:
>                           teardown.throw(exception)

../../env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

item = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(item: Item) -> Generator[None]:
        xfailed = item.stash.get(xfailed_key, None)
        if xfailed is None:
            item.stash[xfailed_key] = xfailed = evaluate_xfail_marks(item)
    
        if xfailed and not item.config.option.runxfail and not xfailed.run:
            xfail("[NOTRUN] " + xfailed.reason)
    
        try:
>           return (yield)
                    ^^^^^

../../env/lib/python3.11/site-packages/_pytest/skipping.py:268: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x7fd9b0cb5510>>]
caller_kwargs = {'item': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = False

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
>                       res = hook_impl.function(*args)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_callers.py:121: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

item = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    def pytest_runtest_call(item: Item) -> None:
        _update_current_test_var(item, "call")
        try:
            del sys.last_type
            del sys.last_value
            del sys.last_traceback
            if sys.version_info >= (3, 12, 0):
                del sys.last_exc  # type:ignore[attr-defined]
        except AttributeError:
            pass
        try:
>           item.runtest()

../../env/lib/python3.11/site-packages/_pytest/runner.py:184: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    def runtest(self) -> None:
        """Execute the underlying test function."""
>       self.ihook.pytest_pyfunc_call(pyfuncitem=self)

../../env/lib/python3.11/site-packages/_pytest/python.py:1707: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <HookCaller 'pytest_pyfunc_call'>
kwargs = {'pyfuncitem': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}

    def __call__(self, **kwargs: object) -> Any:
        """Call the hook.
    
        Only accepts keyword arguments, which should match the hook
        specification.
    
        Returns the result(s) of calling all registered plugins, see
        :ref:`calling`.
        """
        assert not self.is_historic(), (
            "Cannot directly call a historic hook - use call_historic instead."
        )
        self._verify_all_args_are_provided(kwargs)
        firstresult = self.spec.opts.get("firstresult", False) if self.spec else False
        # Copy because plugins may register other plugins during iteration (#438).
>       return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_hooks.py:512: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.config.PytestPluginManager object at 0x7fd9b2b4b490>
hook_name = 'pytest_pyfunc_call'
methods = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/...lugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
kwargs = {'pyfuncitem': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = True

    def _hookexec(
        self,
        hook_name: str,
        methods: Sequence[HookImpl],
        kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        # called from all hookcaller instances.
        # enable_tracing will set its own wrapping function at self._inner_hookexec
>       return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_manager.py:120: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_pyfunc_call'
hook_impls = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/...lugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
caller_kwargs = {'pyfuncitem': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = True

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
                        res = hook_impl.function(*args)
                        if res is not None:
                            results.append(res)
                            if firstresult:  # halt further impl calls
                                break
            except BaseException as exc:
                exception = exc
        finally:
            if firstresult:  # first result hooks return a single value
                result = results[0] if results else None
            else:
                result = results
    
            # run all wrapper post-yield blocks
            for teardown in reversed(teardowns):
                try:
                    if exception is not None:
                        try:
                            teardown.throw(exception)
                        except RuntimeError as re:
                            # StopIteration from generator causes RuntimeError
                            # even for coroutine usage - see #544
                            if (
                                isinstance(exception, StopIteration)
                                and re.__cause__ is exception
                            ):
                                teardown.close()
                                continue
                            else:
                                raise
                    else:
                        teardown.send(result)
                    # Following is unreachable for a well behaved hook wrapper.
                    # Try to force finalizers otherwise postponed till GC action.
                    # Note: close() may raise if generator handles GeneratorExit.
                    teardown.close()
                except StopIteration as si:
                    result = si.value
                    exception = None
                    continue
                except BaseException as e:
                    exception = e
                    continue
                _raise_wrapfail(teardown, "has second yield")
    
        if exception is not None:
>           raise exception

../../env/lib/python3.11/site-packages/pluggy/_callers.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_pyfunc_call'
hook_impls = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/...lugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
caller_kwargs = {'pyfuncitem': <Function test_pickle_dict_of_rows_is_keyed_by_filename>}
firstresult = True

    def _multicall(
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Execute a call into multiple python functions/methods and return the
        result(s).
    
        ``caller_kwargs`` comes from HookCaller.__call__().
        """
        __tracebackhide__ = True
        results: list[object] = []
        exception = None
        try:  # run impl and wrapper setup functions in a loop
            teardowns: list[Teardown] = []
            try:
                for hook_impl in reversed(hook_impls):
                    try:
                        args = [caller_kwargs[argname] for argname in hook_impl.argnames]
                    except KeyError as e:
                        # coverage bug - this is tested
                        for argname in hook_impl.argnames:  # pragma: no cover
                            if argname not in caller_kwargs:
                                raise HookCallError(
                                    f"hook call must provide argument {argname!r}"
                                ) from e
    
                    if hook_impl.hookwrapper:
                        function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
    
                        next(function_gen)  # first yield
                        teardowns.append(function_gen)
    
                    elif hook_impl.wrapper:
                        try:
                            # If this cast is not valid, a type error is raised below,
                            # which is the desired response.
                            res = hook_impl.function(*args)
                            function_gen = cast(Generator[None, object, object], res)
                            next(function_gen)  # first yield
                            teardowns.append(function_gen)
                        except StopIteration:
                            _raise_wrapfail(function_gen, "did not yield")
                    else:
>                       res = hook_impl.function(*args)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/pluggy/_callers.py:121: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

pyfuncitem = <Function test_pickle_dict_of_rows_is_keyed_by_filename>

    @hookimpl(trylast=True)
    def pytest_pyfunc_call(pyfuncitem: Function) -> object | None:
        testfunction = pyfuncitem.obj
        if is_async_function(testfunction):
            async_fail(pyfuncitem.nodeid)
        funcargs = pyfuncitem.funcargs
        testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
>       result = testfunction(**testargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/site-packages/_pytest/python.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

tmp_path = PosixPath('${TMPDIR}/pytest-of-${USER}/pytest-8/test_pickle_dict_of_rows_is_ke0')

    def test_pickle_dict_of_rows_is_keyed_by_filename(tmp_path):
>       pd = pytest.importorskip("pandas")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_bones_terrain_survey.py:119: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

modname = 'pandas', minversion = None, reason = None
exc_type = <class 'ModuleNotFoundError'>

    def importorskip(
        modname: str,
        minversion: str | None = None,
        reason: str | None = None,
        *,
        exc_type: type[ImportError] | None = None,
    ) -> Any:
        """Import and return the requested module ``modname``, or skip the
        current test if the module cannot be imported.
    
        :param modname:
            The name of the module to import.
        :param minversion:
            If given, the imported module's ``__version__`` attribute must be at
            least this minimal version, otherwise the test is still skipped.
        :param reason:
            If given, this reason is shown as the message when the module cannot
            be imported.
        :param exc_type:
            The exception that should be captured in order to skip modules.
            Must be :py:class:`ImportError` or a subclass.
    
            Defaults to :class:`ModuleNotFoundError` when not given, which means
            the module must be missing for the test to be skipped.
            Pass ``exc_type=ImportError`` to also skip modules that raise
            :class:`ImportError` during import.
    
            See :ref:`import-or-skip-import-error` for details.
    
    
        :returns:
            The imported module. This should be assigned to its canonical name.
    
        :raises pytest.skip.Exception:
            If the module cannot be imported.
    
        Example::
    
            docutils = pytest.importorskip("docutils")
    
        .. versionadded:: 8.2
    
            The ``exc_type`` parameter.
    
        .. versionchanged:: 9.1
    
            The default for ``exc_type`` is now :class:`ModuleNotFoundError`.
        """
        import warnings
    
        __tracebackhide__ = True
        compile(modname, "", "eval")  # to catch syntaxerrors
    
        # Keep the public signature compatible while using the pytest 9.1 default behavior.
        if exc_type is None:
            exc_type = ModuleNotFoundError
    
        skipped: Skipped | None = None
    
        with warnings.catch_warnings():
            # Make sure to ignore ImportWarnings that might happen because
            # of existing directories with the same name we're trying to
            # import but without a __init__.py file.
            warnings.simplefilter("ignore")
    
            try:
>               importlib.import_module(modname)

../../env/lib/python3.11/site-packages/_pytest/outcomes.py:264: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'pandas', package = None

    def import_module(name, package=None):
        """Import a module.
    
        The 'package' argument is required when performing a relative import. It
        specifies the package to use as the anchor point from which to resolve the
        relative import to an absolute import.
    
        """
        level = 0
        if name.startswith('.'):
            if not package:
                msg = ("the 'package' argument is required to perform a relative "
                       "import for {!r}")
                raise TypeError(msg.format(name))
            for character in name:
                if character != '.':
                    break
                level += 1
>       return _bootstrap._gcd_import(name[level:], package, level)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

../../env/lib/python3.11/importlib/__init__.py:126: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'pandas', package = None, level = 0

>   ???

<frozen importlib._bootstrap>:1204: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'pandas', import_ = <function _gcd_import at 0x7fd9b2ce3d80>

>   ???

<frozen importlib._bootstrap>:1176: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

name = 'pandas', import_ = <function _gcd_import at 0x7fd9b2ce3d80>

>   ???

<frozen importlib._bootstrap>:1147: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

spec = ModuleSpec(name='pandas', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7fd8c839d2d0>, origin='/data... submodule_search_locations=['${STEP7_ROOT}/runtime_extra/pandas'])

>   ???

<frozen importlib._bootstrap>:690: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_frozen_importlib_external.SourceFileLoader object at 0x7fd8c839d2d0>
module = <module 'pandas' from '${STEP7_ROOT}/runtime_extra/pandas/__init__.py'>

>   ???

<frozen importlib._bootstrap_external>:940: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

f = <built-in function exec>
args = (<code object <module> at 0x555c8cccc840, file "${STEP7_ROOT}/runt...e, '__package__': 'pandas', '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x7fd8c839d2d0>, ...})
kwds = {}

>   ???

<frozen importlib._bootstrap>:241: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    from __future__ import annotations
    
    import os
    import warnings
    
    __docformat__ = "restructuredtext"
    
    # Let users know if they're missing any of our hard dependencies
    _hard_dependencies = ("numpy", "pytz", "dateutil")
    _missing_dependencies = []
    
    for _dependency in _hard_dependencies:
        try:
            __import__(_dependency)
        except ImportError as _e:  # pragma: no cover
            _missing_dependencies.append(f"{_dependency}: {_e}")
    
    if _missing_dependencies:  # pragma: no cover
>       raise ImportError(
            "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
        )
E       ImportError: Unable to import required dependencies:
E       pytz: No module named 'pytz'

../runtime_extra/pandas/__init__.py:19: ImportError
```

</details>

## 3. 执行状态与账本

| 阶段 | 状态 |
| --- | --- |
| 批准版本checkout、环境检查、空闲GPU检查 | 完成 |
| 82项测试门槛 | 失败：81通过、1失败 |
| 默认E集复制及逐文件SHA-256核对 | 未执行，无本轮传输验收总数 |
| 单卡冒烟、resume、双卡冒烟、显存预检 | 未执行 |
| 每卡batch512/256选择 | 尚未产生显存预检结果，未选择 |
| 3,000更新、3小时pilot | 未启动 |
| 实际更新 / 窗口 / 生成帧 | **0 / 0 / 0** |

计划第4节的数值、一帧误差、16帧根漂移、关节穿透、骨盆高度、滑步与支撑代价均**未评估**。没有pilot的`eval.jsonl`或`progress.jsonl`，因此没有训练曲线、逐评估表或六窗口闭环接触图。

原始测试日志和JUnit保留在A100任务仓库的`runs/tmt_pilot_coord/`。本轮通过私有Drive取回的是从JUnit提取的[停止记录](../../examples/results/terrain_controller_pilot_a100_20261024/preflight_report.json)，未传回原始日志或checkpoint。

恢复工作的位置是环境准备：在同一独立依赖目录补齐pytz，并先确认pandas能导入，再重新通过全部82项测试。数据迁移与GPU预检仍在测试门槛之后。

## 4. 发布检查

本次发布停止记录和完整traceback，并从计划与记录索引链接。训练源码保持`9785c01`；发布前运行仓库检查及每份变更Markdown的UTF-8/U+FFFD检查。
