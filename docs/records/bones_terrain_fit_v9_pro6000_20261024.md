# BONES terrain v9：PRO6000 全量拟合与 controller 测试

实际执行日期：**2026-10-25（Asia/Singapore）**。文件名中的 `20261024` 沿用任务约定。执行源码为 `9ad7edddb585cc9588d3909ebdfb9f5e39393db8`，方法见 [第14节](bones_terrain_method_v2_20261023.md#14-v9-修订根据-v8-开发集结果)。

**6,788 条全部拟合成功，外层耗时 829.0 秒；地形目标有一项未通过，controller 有一项测试失败。** 箱体组 `body_rests_lower` 丢弃了 115 块受检抬高足底证据中的 6 块，比例 **5.22%**，超过 1% 门槛。匍匐、手膝、跪姿等其余指定数值目标全部通过。逐图审查还发现楼梯漏建、搬物动作配墙和靠墙动作缺墙，见第7节。

## 1. 执行环境与测试

在 PRO6000 项目独立环境执行，Python **3.11.16**，Torch **2.7.1+cu128**，Kimodo 沿用上一轮 `PYTHONPATH`。`nproc=72`，单个进程池使用 **68 workers**。`CUDA_VISIBLE_DEVICES=''`，OMP、OpenBLAS、MKL 均为单线程。本轮无训练、无参数调整，源 BVH、旧运行结果和 cloth 环境保持原样；主线程使用唯一持久 SSH 并负责 Git。

代码以 `git pull --ff-only` 更新到上述版本；服务器通过已核对的缓存 Git 对象和差量 bundle 完成同一快进。官方 skinner 使用 **18,056 顶点、frame stride 2**。初始化的 limb 顶点数分别为 `foot_left=1258`、`foot_right=1258`、`leg_left=789`、`leg_right=794`。

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$KIMODO_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
PY="$PFNN_PROJECT/env/bin/python"

$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py \
  -q --junitxml=runs/bones_fit_v9_full_coord/terrain_tests.xml
$PY -m pytest tests/test_skeleton_ray_sampler.py \
  -q --junitxml=runs/bones_fit_v9_full_coord/ray_tests.xml
$PY -m pytest tests/test_terrain_controller.py -q --full-trace --color=no \
  --junitxml=runs/bones_fit_v9_full_coord/controller_tests.xml
```

| 测试套件 | 实际输出末行 | 跳过 | 日志 |
| --- | --- | ---: | --- |
| 四个地形测试文件 | `60 passed in 4.30s` | 0 | [完整日志](../../examples/results/bones_terrain_fit_v9_20261024/tests/terrain_tests.txt) |
| 骨骼射线 | `6 passed in 1.02s` | 0 | [完整日志](../../examples/results/bones_terrain_fit_v9_20261024/tests/ray_tests.txt) |
| Controller | `1 failed, 8 passed, 4 warnings in 3.98s` | 0 | [完整 traceback](../../examples/results/bones_terrain_fit_v9_20261024/tests/controller_tests.txt) |

Controller 的失败点为 [test_terrain_controller.py:60](../../tests/test_terrain_controller.py#L60) 中 `test_torch_sdf_matches_reference_and_pushes_points_out` 的第一次距离对照：`np.allclose(..., atol=1e-9)` 返回 False。输出可见 Torch 外部距离 `0.14129053`、NumPy `0.14129153` 等约 `1e-6` 差异；后面的梯度推出断言尚未执行。按任务约定保留完整错误、继续地形运行，没有修改 controller 或拟合实现。

<details>
<summary>展开 controller 测试的完整输出和 traceback（仅将私有目录替换为角色变量）</summary>

```text
...F.....                                                                [100%]
=================================== FAILURES ===================================
____________ test_torch_sdf_matches_reference_and_pushes_points_out ____________

cls = <class '_pytest.runner.CallInfo'>
func = <function call_and_report.<locals>.<lambda> at 0x75d6af9631a0>
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/runner.py:361: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>       lambda: runtest_hook(item=item, **kwds),
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        when=when,
        reraise=get_reraise_exceptions(item.config),
    )

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/runner.py:250: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <HookCaller 'pytest_runtest_call'>
kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}

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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_hooks.py:512: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.config.PytestPluginManager object at 0x75d6b41803d0>
hook_name = 'pytest_runtest_call'
methods = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_manager.py:120: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
caller_kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
caller_kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>
item = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: nodes.Item) -> Generator[None]:
        self.log_cli_handler.set_when("call")
    
        with self._runtest_for(item, "call"):
>           yield

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/logging.py:865: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
caller_kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <CaptureManager _method='fd' _global_capturing=<MultiCapture out=<FDCapture 1 oldfd=5 _state='suspended' tmpfile=<_io....xtIOWrapper name='/dev/null' mode='r' encoding='utf-8'>> _state='suspended' _in_suspended=False> _capture_fixture=None>
item = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None]:
        with self.item_capture("call", item):
>           return (yield)
                    ^^^^^

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/capture.py:900: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
caller_kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

item = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/skipping.py:268: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_runtest_call'
hook_impls = [<HookImpl plugin_name='threadexception', plugin=<module '_pytest.threadexception' from '${PFNN_PROJECT}...xture=None>>, <HookImpl plugin_name='logging-plugin', plugin=<_pytest.logging.LoggingPlugin object at 0x75d6b2da5310>>]
caller_kwargs = {'item': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:121: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

item = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/runner.py:184: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

    def runtest(self) -> None:
        """Execute the underlying test function."""
>       self.ihook.pytest_pyfunc_call(pyfuncitem=self)

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/python.py:1707: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <HookCaller 'pytest_pyfunc_call'>
kwargs = {'pyfuncitem': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}

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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_hooks.py:512: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <_pytest.config.PytestPluginManager object at 0x75d6b41803d0>
hook_name = 'pytest_pyfunc_call'
methods = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/en..._plugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
kwargs = {'pyfuncitem': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_manager.py:120: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_pyfunc_call'
hook_impls = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/en..._plugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
caller_kwargs = {'pyfuncitem': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

hook_name = 'pytest_pyfunc_call'
hook_impls = [<HookImpl plugin_name='python', plugin=<module '_pytest.python' from '${PFNN_PROJECT}/en..._plugin' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/anyio/pytest_plugin.py'>>]
caller_kwargs = {'pyfuncitem': <Function test_torch_sdf_matches_reference_and_pushes_points_out>}
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

${PFNN_PROJECT}/env/lib/python3.11/site-packages/pluggy/_callers.py:121: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

pyfuncitem = <Function test_torch_sdf_matches_reference_and_pushes_points_out>

    @hookimpl(trylast=True)
    def pytest_pyfunc_call(pyfuncitem: Function) -> object | None:
        testfunction = pyfuncitem.obj
        if is_async_function(testfunction):
            async_fail(pyfuncitem.nodeid)
        funcargs = pyfuncitem.funcargs
        testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
>       result = testfunction(**testargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^

${PFNN_PROJECT}/env/lib/python3.11/site-packages/_pytest/python.py:167: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def test_torch_sdf_matches_reference_and_pushes_points_out():
        torch = pytest.importorskip("torch")
        from motionbricks.controller.scene_sdf import box_sdf, penetration_cost
        rng = np.random.default_rng(0)
        boxes_list = [GROUND] + [box(rng.uniform(-1, 1, 3), rng.uniform(.1, .4, 3), rng.uniform(0, 3)) for _ in range(5)]
        packed = pack_boxes(boxes_list, 8)
        tensors = tuple(torch.as_tensor(a, dtype=torch.float64 if a.dtype != bool else torch.bool) for a in packed)
        points = rng.uniform(-1.2, 1.2, (2, 50, 3))
>       assert np.allclose(box_sdf(torch.as_tensor(points), tensors).numpy(), box_sdf_numpy(points, packed), atol=1e-9)
E       AssertionError: assert False
E        +  where False = <function allclose at 0x75d6b2d27770>(array([[-0.03399514,  0.14129053,  0.22631907,  0.73462791,  0.15975985,\n        -0.03074726,  0.19331805,  0.68283857...6745,  0.17748029,  0.05573634,  0.54251686,\n        -0.03720254,  0.26304121,  0.20666942,  0.83688409,  0.33795967]]), array([[-0.03399514,  0.14129153,  0.22632007,  0.73462891,  0.15976085,\n        -0.03074726,  0.19331905,  0.68283957...6845,  0.17748129,  0.05573734,  0.54251786,\n        -0.03720254,  0.26304221,  0.20667042,  0.83688509,  0.33796067]]), atol=1e-09)
E        +    where <function allclose at 0x75d6b2d27770> = np.allclose
E        +    and   array([[-0.03399514,  0.14129053,  0.22631907,  0.73462791,  0.15975985,\n        -0.03074726,  0.19331805,  0.68283857...6745,  0.17748029,  0.05573634,  0.54251686,\n        -0.03720254,  0.26304121,  0.20666942,  0.83688409,  0.33795967]]) = <built-in method numpy of Tensor object at 0x75d5aa4eb710>()
E        +      where <built-in method numpy of Tensor object at 0x75d5aa4eb710> = tensor([[-0.0340,  0.1413,  0.2263,  0.7346,  0.1598, -0.0307,  0.1933,  0.6828,\n          0.0291,  0.1928,  0.1028,  ....0112,  0.1587,  0.1775,  0.0557,  0.5425, -0.0372,  0.2630,  0.2067,\n          0.8369,  0.3380]], dtype=torch.float64).numpy
E        +        where tensor([[-0.0340,  0.1413,  0.2263,  0.7346,  0.1598, -0.0307,  0.1933,  0.6828,\n          0.0291,  0.1928,  0.1028,  ....0112,  0.1587,  0.1775,  0.0557,  0.5425, -0.0372,  0.2630,  0.2067,\n          0.8369,  0.3380]], dtype=torch.float64) = <function box_sdf at 0x75d6af913ba0>(tensor([[[-4.5542e-01, -3.3995e-02,  9.3477e-01],\n         [ 1.0417e+00, -3.4129e-01,  1.7167e-01],\n         [-4.2751e...      [-4.0825e-01,  1.0648e+00, -7.2168e-01],\n         [ 2.9217e-02, -1.1424e+00, -8.0792e-01]]], dtype=torch.float64), (tensor([[ 0.0000e+00, -1.0000e-01,  0.0000e+00],\n        [ 2.7392e-01, -4.6043e-01, -9.1805e-01],\n        [ 4.5899e-0... [ 0.0000,  0.0000,  1.0000]]], dtype=torch.float64), tensor([ True,  True,  True,  True,  True,  True, False, False])))
E        +          where tensor([[[-4.5542e-01, -3.3995e-02,  9.3477e-01],\n         [ 1.0417e+00, -3.4129e-01,  1.7167e-01],\n         [-4.2751e...      [-4.0825e-01,  1.0648e+00, -7.2168e-01],\n         [ 2.9217e-02, -1.1424e+00, -8.0792e-01]]], dtype=torch.float64) = <built-in method as_tensor of type object at 0x75d6a23a8fa0>(array([[[-4.55419499e-01, -3.39951388e-02,  9.34770802e-01],\n        [ 1.04170444e+00, -3.41291528e-01,  1.71671594e-0...    [-4.08250002e-01,  1.06482664e+00, -7.21683982e-01],\n        [ 2.92167788e-02, -1.14236832e+00, -8.07916581e-01]]]))
E        +            where <built-in method as_tensor of type object at 0x75d6a23a8fa0> = <module 'torch' from '${PFNN_PROJECT}/env/lib/python3.11/site-packages/torch/__init__.py'>.as_tensor
E        +    and   array([[-0.03399514,  0.14129153,  0.22632007,  0.73462891,  0.15976085,\n        -0.03074726,  0.19331905,  0.68283957...6845,  0.17748129,  0.05573734,  0.54251786,\n        -0.03720254,  0.26304221,  0.20667042,  0.83688509,  0.33796067]]) = box_sdf_numpy(array([[[-4.55419499e-01, -3.39951388e-02,  9.34770802e-01],\n        [ 1.04170444e+00, -3.41291528e-01,  1.71671594e-0...    [-4.08250002e-01,  1.06482664e+00, -7.21683982e-01],\n        [ 2.92167788e-02, -1.14236832e+00, -8.07916581e-01]]]), (array([[ 0.00000000e+00, -1.00000000e-01,  0.00000000e+00],\n       [ 2.73923375e-01, -4.60426572e-01, -9.18052952e-01...],\n        [ 0.        ,  0.        ,  1.        ]]]), array([ True,  True,  True,  True,  True,  True, False, False])))

tests/test_terrain_controller.py:60: AssertionError
=============================== warnings summary ===============================
tests/test_terrain_controller.py::test_untrained_transformer_holds_the_pose_and_has_fixed_shapes
  ${PFNN_PROJECT}/env/lib/python3.11/site-packages/lightning_fabric/plugins/environments/xla.py:18: DeprecationWarning: `ModuleAvailableCache` is a special case of `RequirementCache`. Please use `RequirementCache(module=...)` instead.
    from lightning_fabric.accelerators.tpu import _XLA_AVAILABLE, TPUAccelerator

tests/test_terrain_controller.py::test_untrained_transformer_holds_the_pose_and_has_fixed_shapes
  ${PFNN_PROJECT}/env/lib/python3.11/site-packages/lightning_fabric/__init__.py:36: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
    __import__("pkg_resources").declare_namespace(__name__)

tests/test_terrain_controller.py::test_untrained_transformer_holds_the_pose_and_has_fixed_shapes
  ${PFNN_PROJECT}/env/lib/python3.11/site-packages/lightning_fabric/__init__.py:36: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('lightning_fabric')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    __import__("pkg_resources").declare_namespace(__name__)

tests/test_terrain_controller.py::test_untrained_transformer_holds_the_pose_and_has_fixed_shapes
  ${PFNN_PROJECT}/env/lib/python3.11/site-packages/pytorch_lightning/__init__.py:36: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('pytorch_lightning')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    __import__("pkg_resources").declare_namespace(__name__)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_terrain_controller.py::test_torch_sdf_matches_reference_and_pushes_points_out
1 failed, 8 passed, 4 warnings in 3.98s
```

</details>

## 2. 修复开发集记录的乱码

从服务器保存的 `examples/results/bones_terrain_fit_v9_dev_20261024/report.json` 恢复 [v9 开发集记录](bones_terrain_fit_v9_dev_pro6000_20261024.md) 的全部 **60 条**审查文字，逐条按文件名对应。原 Markdown 中有 **1,608 个 U+FFFD**，恢复后为 **0**；原始图片与数值不变。此次记录、对照表与结果文字均以 UTF-8 写入，并在发布前逐文件执行同一检查：

```bash
python -X utf8 -c "import sys;t=open(sys.argv[1],encoding='utf-8').read();assert '\ufffd' not in t" <record.md>
```

## 3. 全量运行与耗时

```bash
$PY -u scripts/fit_bones_terrain_support.py \
  --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v9_full \
  --motion-cache runs/bones_motion_cache_s2 \
  --viz-per-stem 1 --worst-per-group 5 --sample-per-group 5 \
  --variants 4 --skin-stride 2 --workers 68
```

监督器记录退出码 **0**；脚本耗时 **828.1 秒**，外层壁钟 **828.994 秒**。6,788 条拟合、11 个组，拟合失败 **0**，原运行 PNG 渲染 **118/118 成功**，无 GIF。复用 593 条缓存，新增转换 6,195 条；结束时缓存目录有 **6,788 个**对应 NPZ。每个文件名及所属组均与 v7 的 6,788 行逐一核对。

`stage_seconds` 是每条动作在各阶段消耗时间的汇总；多个 worker 同时工作，因此累计秒数大于外层壁钟。P50/P95 表示单条动作的阶段耗时。v7 没有记录同口径分阶段耗时，对照中为 `null`。

| 阶段 | 片段数 | 累计秒 | 单条P50秒 | 单条P95秒 |
| --- | --- | --- | --- | --- |
| motion_cached | 593 | 1612.8 | 2.4637 | 5.9134 |
| motion_converted | 6195 | 27530.6 | 3.826 | 9.0984 |
| scene | 6788 | 15863.3 | 1.625 | 6.7806 |
| screen | 6788 | 342.2 | 0.0396 | 0.1101 |
| support_fit | 6788 | 7440.4 | 0.7711 | 2.788 |

各组累计阶段秒数如下；单条动作的 P50/P95 同时保存在 [完整对照 JSON](../../examples/results/bones_terrain_fit_v9_20261024/comparison.json)。

| 组 | motion_cached | motion_converted | screen | support_fit | scene |
| --- | --- | --- | --- | --- | --- |
| box_support | 68.7 | 20.7 | 2.6 | 58.0 | 68.2 |
| crouch | 99.7 | 4208.0 | 61.8 | 804.1 | 2091.5 |
| flat_control | 114.3 | 4515.7 | 66.6 | 977.8 | 1948.7 |
| kneel | 131.9 | 3320.0 | 29.2 | 821.4 | 2526.7 |
| known_dimension_support | 230.8 | 4082.1 | 69.1 | 1496.2 | 1866.3 |
| ladder | 117.2 | 94.8 | 4.0 | 97.4 | 148.9 |
| obstacle_clearance | 95.2 | 194.9 | 4.5 | 99.9 | 128.6 |
| prone | 263.1 | 4732.6 | 38.0 | 1197.3 | 2744.4 |
| quadruped | 278.5 | 4824.5 | 38.4 | 1316.8 | 3160.1 |
| stairs | 95.2 | 46.4 | 3.1 | 215.0 | 80.8 |
| wall_contact | 118.2 | 1490.8 | 24.9 | 356.6 | 1099.0 |

## 4. v7 / v9 全量比较

完整 [Markdown 对照](../../examples/results/bones_terrain_fit_v9_20261024/comparison.md)、[TSV](../../examples/results/bones_terrain_fit_v9_20261024/comparison.tsv)、[JSON](../../examples/results/bones_terrain_fit_v9_20261024/comparison.json) 保留 v7 的全部字段：最小/最大支撑指标、地面、flags、足底 patch、悬空板、墙与顶棚、变体接受情况、拒绝原因、资格和排除前资格，并添加 `stage_seconds`。v8 仅运行开发集，本轮直接比较相同 6,788 条的 v7 与 v9。

F 为四个场景变体中 `foot_evidence_not_supported` 的命中总次数；四箱指最小支撑中至少四个箱体，不含地面。E 为环境可解释，G 为仅几何一致且低姿态未获解释，R 为拒绝，X 为未建模组排除。`body_rests_lower` 计被该原因丢弃的抬高足底 patch。

| 组 | 片段 | 最大支撑超2cm v7→v9 | F v7→v9 | 四箱片段 v7→v9 | 身体低于足底丢弃 v7→v9 | 墙 v7→v9 |
| --- | --- | --- | --- | --- | --- | --- |
| box_support | 48 | 0 → 0 | 32 → 24 | 3 → 0 | 14 → 6 | 0 → 0 |
| crouch | 1143 | 0 → 0 | 19 → 0 | 4 → 0 | 3 → 0 | 55 → 55 |
| flat_control | 900 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 3 → 3 |
| kneel | 638 | 0 → 0 | 140 → 4 | 0 → 0 | 468 → 137 | 2 → 2 |
| known_dimension_support | 1760 | 0 → 0 | 172 → 196 | 0 → 0 | 82 → 0 | 50 → 50 |
| ladder | 64 | 0 → 0 | 32 → 40 | 0 → 0 | 12 → 0 | 3 → 3 |
| obstacle_clearance | 86 | 0 → 0 | 36 → 28 | 0 → 0 | 3 → 0 | 0 → 0 |
| prone | 919 | 0 → 0 | 582 → 98 | 150 → 13 | 6750 → 1392 | 0 → 0 |
| quadruped | 831 | 0 → 0 | 436 → 0 | 21 → 0 | 4075 → 1875 | 33 → 33 |
| stairs | 49 | 0 → 0 | 60 → 60 | 25 → 25 | 3 → 0 | 0 → 0 |
| wall_contact | 350 | 0 → 0 | 8 → 0 | 0 → 0 | 41 → 9 | 106 → 106 |

| 组 | v7 E/G/R/X | v9 E/G/R/X |
| --- | --- | --- |
| box_support | 9/31/8/0 | 9/33/6/0 |
| crouch | 1109/30/4/0 | 1111/32/0/0 |
| flat_control | 900/0/0/0 | 900/0/0/0 |
| kneel | 265/338/35/0 | 272/365/1/0 |
| known_dimension_support | 1695/22/43/0 | 1692/19/49/0 |
| ladder | 0/0/0/64 | 0/0/0/64 |
| obstacle_clearance | 0/0/0/86 | 0/0/0/86 |
| prone | 549/226/144/0 | 637/258/24/0 |
| quadruped | 636/86/109/0 | 684/147/0/0 |
| stairs | 27/7/15/0 | 27/7/15/0 |
| wall_contact | 316/32/2/0 | 318/32/0/0 |

已知尺寸组的 F 从 **172 增至 196**，虽然该组的 `body_rests_lower` 已降为零，仍有其他支撑覆盖缺口。第6节列出的20条均保留一个箱体，同时带有 `evidence_reduced_for_body_conflict` 标记，四个场景变体均拒绝。

## 5. 目标逐项判定

| 目标 | 实测 | 要求 | 判定 |
| --- | --- | --- | --- |
| 所有组最大支撑超2cm | 0 / 6,788条 | 0条 | 通过 |
| prone F / 四箱片段 | 98 / 13 | ≤544 / ≤38 | 通过 |
| quadruped F / 四箱片段 | 0 / 0 | ≤662 / ≤8 | 通过 |
| kneel F | 4 | ≤140 | 通过 |
| known_dimension_support身体低于足底丢弃 | 0 / 3,613 = 0% | ≤1% | 通过 |
| box_support身体低于足底丢弃 | 6 / 115 = 5.22% | ≤1% | **失败** |
| stairs身体低于足底丢弃 | 0 / 242 = 0% | ≤1% | 通过 |
| ladder全部排除 | 64 / 64 | 100% | 通过 |
| obstacle_clearance全部排除 | 86 / 86 | 100% | 通过 |
| crouch墙 | 55 | ≤55 | 通过 |
| wall_contact覆盖 | 106 / 350 = 30.29% | ≥25% | 通过 |
| flat_control E | 900 / 900 = 100% | 100% | 通过 |

正常三组的丢弃比例分母为本轮接受检查的抬高足底 patch 数，与 v7 表的统计口径一致。箱体组的6块证据来自以下4条动作；该组目标判为失败，未修改阈值或补跑。

| 文件名 | 丢弃patch | 保留箱 | 侧别:离地高度m | body_below_fraction | body_drop_median_m |
| --- | --- | --- | --- | --- | --- |
| jump_form_box_to_safety_roll_180_R_001__A501 | 1 | 1 | left:0.1324 | 1.0000 | 0.1261 |
| jump_form_box_to_safety_roll_360_R_001__A501 | 2 | 1 | right:0.1275, right:0.1061 | 1.0000, 0.6667 | 0.0706, 0.0550 |
| jump_to_shoulder_roll_R_001__A305 | 1 | 1 | right:0.1108 | 0.6000 | 0.0765 |
| jump_to_shoulder_roll_R_002__A305 | 2 | 0 | left:0.1076, right:0.1149 | 0.5000, 0.5714 | -0.0153, 0.0614 |

完整的侧别、面积、身体高度比例和所有丢弃原因保存在 [box_body_drop_details.json](../../examples/results/bones_terrain_fit_v9_20261024/box_body_drop_details.json)。

## 6. 已知尺寸组 F 最高的20条

按 F 降序、文件名字典序打破并列，得到以下20条，F 均为4、资格均为R。每条的PNG和viewer都随结果发布。补齐缺失图像时只读取已有 fit 与缓存关节，以4个CPU worker渲染，**没有重新拟合**。

| 文件名 / PNG | F | 最小/最大箱 | 资格 | Viewer |
| --- | --- | --- | --- | --- |
| [come_down_50cm_box_R_001__A103](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A103.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_001__A103.json) |
| [come_down_50cm_box_R_001__A277](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A277.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_001__A277.json) |
| [come_down_50cm_box_R_001__A279](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A279.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_001__A279.json) |
| [come_down_50cm_box_R_001__A292](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A292.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_001__A292.json) |
| [come_down_50cm_box_R_001__A373](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A373.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_001__A373.json) |
| [come_down_50cm_box_R_002__A137](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A137.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A137.json) |
| [come_down_50cm_box_R_002__A223](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A223.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A223.json) |
| [come_down_50cm_box_R_002__A277](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A277.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A277.json) |
| [come_down_50cm_box_R_002__A287](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A287.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A287.json) |
| [come_down_50cm_box_R_002__A288](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A288.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A288.json) |
| [come_down_50cm_box_R_002__A328](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A328.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A328.json) |
| [come_down_50cm_box_R_002__A373](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A373.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A373.json) |
| [come_down_50cm_box_R_002__A410](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A410.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_002__A410.json) |
| [come_down_50cm_box_R_003__A087](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A087.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A087.json) |
| [come_down_50cm_box_R_003__A090](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A090.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A090.json) |
| [come_down_50cm_box_R_003__A174](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A174.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A174.json) |
| [come_down_50cm_box_R_003__A187](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A187.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A187.json) |
| [come_down_50cm_box_R_003__A199](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A199.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A199.json) |
| [come_down_50cm_box_R_003__A223](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A223.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A223.json) |
| [come_down_50cm_box_R_003__A224](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A224.png) | 4 | 1/1 | R | [JSON](../../examples/results/bones_terrain_fit_v9_20261024/viewer/come_down_50cm_box_R_003__A224.json) |

## 7. 逐图审查

审查集合为11组各5张固定哈希抽样，共55张，加上上述20张F样本，共 **75 张**；两部分没有重叠。`--worst-per-group 5` 的代码只在最小支撑存在超2cm帧时添加 worst 图，本轮没有符合条件的片段，因此额外 worst 图为0。其余每词干抽样图保留在服务器。

全部75张图的3D和侧视面板均已查看，检查时以完整四图拼页展示，没有裁掉面板。下表每条对应一张原图；这次图审也单独保留了数值分类未捕捉到的问题：下楼漏建、搬物配墙、靠墙缺墙。PNG中的蓝色人体轨迹来自源动作。

| 序号 | 文件名 / PNG | 逐图判断 |
| --- | --- | --- |
| 1 | [avoid_obstacle_jump_run_ff_180_R_002__A502](../../examples/results/bones_terrain_fit_v9_20261024/viz/obstacle_clearance/avoid_obstacle_jump_run_ff_180_R_002__A502.png) | 跑跳轨迹只配连续地面，腾空处没有障碍物，保持X排除。 |
| 2 | [box_jump_001__A361](../../examples/results/bones_terrain_fit_v9_20261024/viz/box_support/box_jump_001__A361.png) | 约0.55m的单个箱体对应跳上后的足部位置，落点有支撑，屈膝动作未获顶棚解释。 |
| 3 | [box_jump_down_360_002__A359](../../examples/results/bones_terrain_fit_v9_20261024/viz/box_support/box_jump_down_360_002__A359.png) | 约0.58m的起跳箱与离箱轨迹对应，落地侧没有额外小块，保留G分类。 |
| 4 | [box_jump_down_360_002__A361](../../examples/results/bones_terrain_fit_v9_20261024/viz/box_support/box_jump_down_360_002__A361.png) | 约0.56m的平台托住起跳位置，身体落向旁侧地面，未生成多余墙体或顶棚。 |
| 5 | [come_down_50cm_box_R_001__A103](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A103.png) | 下箱起点仍有约0.59m的窄支撑，箱面范围收缩，标有身体冲突缩减证据，F=4。 |
| 6 | [come_down_50cm_box_R_001__A277](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A277.png) | 起点保留约0.58m箱体，但实体只覆盖较窄的足部区域，缺支撑检查仍在四个变体中失败。 |
| 7 | [come_down_50cm_box_R_001__A279](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A279.png) | 约0.53m平台位于后脚下，横向宽度很窄；下箱轨迹保留，F=4。 |
| 8 | [come_down_50cm_box_R_001__A292](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A292.png) | 约0.60m高层没有消失，箱体被收成窄柱，记录有证据缩减和四次缺支撑拒绝。 |
| 9 | [come_down_50cm_box_R_001__A371](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A371.png) | 下箱前的双足区域有较宽的箱面，约0.55m高度与起始脚位对应，落地侧清空。 |
| 10 | [come_down_50cm_box_R_001__A373](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A373.png) | 约0.59m的窄箱托住起点局部，跨步到地面时没有附加几何，足底覆盖检查仍失败。 |
| 11 | [come_down_50cm_box_R_002__A137](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A137.png) | 约0.50m起始平台存在，但最小实体只留窄幅，身体冲突缩减后仍有F=4。 |
| 12 | [come_down_50cm_box_R_002__A223](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A223.png) | 约0.61m箱体与下箱高度对应，保留范围较窄，四个场景变体均报缺支撑。 |
| 13 | [come_down_50cm_box_R_002__A277](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A277.png) | 下箱起点约0.58m支撑仍在，最小箱体收成窄柱，未恢复完整足底证据覆盖。 |
| 14 | [come_down_50cm_box_R_002__A287](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A287.png) | 约0.59m箱面对应后脚高度，实体横向范围很窄，缺支撑拒绝仍为四次。 |
| 15 | [come_down_50cm_box_R_002__A288](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A288.png) | 约0.61m高层只留下单个窄箱，前腿下降路径清空，足底支撑覆盖不足。 |
| 16 | [come_down_50cm_box_R_002__A328](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A328.png) | 约0.57m平台仍位于初始脚下，保留面偏窄，记录有身体冲突缩减和F=4。 |
| 17 | [come_down_50cm_box_R_002__A373](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A373.png) | 约0.58m箱体保住了高处起点，但最小支撑仅为窄柱，四个变体仍拒绝。 |
| 18 | [come_down_50cm_box_R_002__A410](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A410.png) | 约0.60m支撑与起始脚位对应，箱体宽度收缩，落地一侧空旷，F=4。 |
| 19 | [come_down_50cm_box_R_003__A045](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A045.png) | 约0.63m平台覆盖起始双足区域，单步下箱到平地的关系清楚，没有额外墙体。 |
| 20 | [come_down_50cm_box_R_003__A087](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A087.png) | 约0.61m箱体保留在高处脚位下，最小实体较窄，身体冲突缩减后仍缺支撑。 |
| 21 | [come_down_50cm_box_R_003__A090](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A090.png) | 约0.59m高层未丢失，窄箱只覆盖起点局部，四个场景变体均报F。 |
| 22 | [come_down_50cm_box_R_003__A174](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A174.png) | 约0.60m窄支撑托住高处起点，落地脚在箱外；支撑面覆盖检查仍未通过。 |
| 23 | [come_down_50cm_box_R_003__A187](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A187.png) | 约0.59m平台收成窄柱，后脚下方有实体但范围不足，F=4。 |
| 24 | [come_down_50cm_box_R_003__A199](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A199.png) | 约0.59m箱面与下箱起始高度相符，横向实体很窄，保留证据缩减标记。 |
| 25 | [come_down_50cm_box_R_003__A223](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A223.png) | 约0.61m支撑仍在初始足部下方，前脚向箱外落地，四个变体均缺支撑。 |
| 26 | [come_down_50cm_box_R_003__A224](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A224.png) | 约0.57m起始箱体保留，实体范围较窄，下箱后地面清空，F=4。 |
| 27 | [come_up_50cm_box_R_001__A206](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_up_50cm_box_R_001__A206.png) | 约0.61m单箱对应上箱后的足部高度，起步地面与终点平台关系清楚。 |
| 28 | [come_up_50cm_box_R_002__A282](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/come_up_50cm_box_R_002__A282.png) | 约0.59m平台接住上箱后的双脚，沿途没有碎片支撑或附加墙体。 |
| 29 | [crawl_ff_loop_270_R_001__A216](../../examples/results/bones_terrain_fit_v9_20261024/viz/prone/crawl_ff_loop_270_R_001__A216.png) | 横向匍匐全程只有地面和连续低顶棚，脚、膝下没有小箱。 |
| 30 | [crawl_ff_loop_315_005__A131](../../examples/results/bones_terrain_fit_v9_20261024/viz/prone/crawl_ff_loop_315_005__A131.png) | 斜向匍匐路径由低顶棚连续覆盖，身体下方没有离散支撑块。 |
| 31 | [crawl_ff_stop_180_R_001__A230](../../examples/results/bones_terrain_fit_v9_20261024/viz/prone/crawl_ff_stop_180_R_001__A230.png) | 匍匐停止段贴近连续地面，低顶棚覆盖路径，脚下没有残余小柱。 |
| 32 | [crawl_ff_stop_270_R_003__A216](../../examples/results/bones_terrain_fit_v9_20261024/viz/prone/crawl_ff_stop_270_R_003__A216.png) | 侧向匍匐停止下方无碎片箱，低顶棚与身体活动范围对应。 |
| 33 | [crouch_ff_loop_270_R_003__A246](../../examples/results/bones_terrain_fit_v9_20261024/viz/crouch/crouch_ff_loop_270_R_003__A246.png) | 蹲行路径上方有连续顶棚，地面平整，未生成侧墙或脚下小块。 |
| 34 | [crouch_ff_start_315_003__A148](../../examples/results/bones_terrain_fit_v9_20261024/viz/crouch/crouch_ff_start_315_003__A148.png) | 斜向蹲行起步由连续顶棚覆盖，足部下方保持单一地面。 |
| 35 | [crouch_ff_start_360_R_003__A197](../../examples/results/bones_terrain_fit_v9_20261024/viz/crouch/crouch_ff_start_360_R_003__A197.png) | 图中持续弯曲行进，但没有拟合顶棚，状态为no_sustained_low_posture且归E。 |
| 36 | [h_b_w_crouch_to_stand_270_start_001__A344](../../examples/results/bones_terrain_fit_v9_20261024/viz/wall_contact/h_b_w_crouch_to_stand_270_start_001__A344.png) | 蹲起轨迹只有地面，没有靠墙几何或顶棚，保留G分类。 |
| 37 | [h_b_w_crouch_to_stand_270_stop_001__A344](../../examples/results/bones_terrain_fit_v9_20261024/viz/wall_contact/h_b_w_crouch_to_stand_270_stop_001__A344.png) | 站蹲变换区域没有墙和顶棚，靠墙关系未体现，当前为G。 |
| 38 | [h_b_w_stand_to_crouch_270_start_003__A344](../../examples/results/bones_terrain_fit_v9_20261024/viz/wall_contact/h_b_w_stand_to_crouch_270_start_003__A344.png) | 下蹲路径上方有顶棚，背后没有墙，E分类没有补足靠墙几何。 |
| 39 | [idle_crawl_to_idle_crawl_right_R_003__A232](../../examples/results/bones_terrain_fit_v9_20261024/viz/prone/idle_crawl_to_idle_crawl_right_R_003__A232.png) | 伏地转向区域由低顶棚覆盖，身体下方没有小箱。 |
| 40 | [idle_to_wall_leaning_idle_360_R_002__A289](../../examples/results/bones_terrain_fit_v9_20261024/viz/wall_contact/idle_to_wall_leaning_idle_360_R_002__A289.png) | 站立转靠墙动作旁保留一面墙，墙位于躯干后侧，位置与倚靠关系基本相符。 |
| 41 | [idle_turn_360_001__A066](../../examples/results/bones_terrain_fit_v9_20261024/viz/flat_control/idle_turn_360_001__A066.png) | 平地原地转身只有连续地面，没有额外台阶、墙或顶棚。 |
| 42 | [item_pick_up_crouch_walk_R_003__A412](../../examples/results/bones_terrain_fit_v9_20261024/viz/crouch/item_pick_up_crouch_walk_R_003__A412.png) | 取物后蹲行路径有连续顶棚，地面上没有碎片支撑。 |
| 43 | [jog_arc_cw_stop_R_005__A216](../../examples/results/bones_terrain_fit_v9_20261024/viz/flat_control/jog_arc_cw_stop_R_005__A216.png) | 弧线跑动到停止沿平地展开，转弯处没有新增箱体或墙面。 |
| 44 | [jog_backward_start_001__A038](../../examples/results/bones_terrain_fit_v9_20261024/viz/flat_control/jog_backward_start_001__A038.png) | 后退跑起步保持连续平地，脚下无箱体，头顶无附加几何。 |
| 45 | [jog_sideway_135_start_001__A031](../../examples/results/bones_terrain_fit_v9_20261024/viz/flat_control/jog_sideway_135_start_001__A031.png) | 侧向跑起步只配地面，行进方向上的空间保持空旷。 |
| 46 | [jump_over_obstacle_0_75m_R_102__A416](../../examples/results/bones_terrain_fit_v9_20261024/viz/obstacle_clearance/jump_over_obstacle_0_75m_R_102__A416.png) | 0.75m越障动作的腾空段下没有障碍实体，保持X排除。 |
| 47 | [jump_over_obstacle_1_5m_L_101__A416](../../examples/results/bones_terrain_fit_v9_20261024/viz/obstacle_clearance/jump_over_obstacle_1_5m_L_101__A416.png) | 1.5m越障轨迹明显抬高，但场景没有对应障碍体，保持X。 |
| 48 | [jump_over_obstacle_1m_R_102__A305](../../examples/results/bones_terrain_fit_v9_20261024/viz/obstacle_clearance/jump_over_obstacle_1m_R_102__A305.png) | 1m越障路径仅配平地，没有跨越物几何，保持X。 |
| 49 | [jump_over_obstacle_1m_R_104__A305](../../examples/results/bones_terrain_fit_v9_20261024/viz/obstacle_clearance/jump_over_obstacle_1m_R_104__A305.png) | 另一条1m越障样本同样只有地面，障碍未建模，保持X。 |
| 50 | [jump_to_shoulder_roll_R_001__A305](../../examples/results/bones_terrain_fit_v9_20261024/viz/box_support/jump_to_shoulder_roll_R_001__A305.png) | 肩滚前的高处起点只剩窄柱，后续翻滚区域没有低矮碎块，当前为G。 |
| 51 | [jump_to_shoulder_roll_R_102__A416](../../examples/results/bones_terrain_fit_v9_20261024/viz/box_support/jump_to_shoulder_roll_R_102__A416.png) | 约1.11m起跳箱位于初始脚下，后续肩滚落在连续地面，没有额外小箱。 |
| 52 | [kneeling_loop_001__A101](../../examples/results/bones_terrain_fit_v9_20261024/viz/kneel/kneeling_loop_001__A101.png) | 跪姿保持段下方无小箱，头顶有连续顶棚覆盖跪姿范围。 |
| 53 | [kneeling_start_R_003__A408](../../examples/results/bones_terrain_fit_v9_20261024/viz/kneel/kneeling_start_R_003__A408.png) | 站立转跪姿过程中脚、膝下无碎片支撑，没有强加顶棚，保留G。 |
| 54 | [kneeling_stop_002__A037](../../examples/results/bones_terrain_fit_v9_20261024/viz/kneel/kneeling_stop_002__A037.png) | 跪姿起身保持空旷地面，脚和膝下均无箱体，原地变换保留G。 |
| 55 | [kneeling_stop_003__A051](../../examples/results/bones_terrain_fit_v9_20261024/viz/kneel/kneeling_stop_003__A051.png) | 跪姿到站立的垂直活动范围清空，未拟合顶棚或支撑小块。 |
| 56 | [kneeling_stop_003__A097](../../examples/results/bones_terrain_fit_v9_20261024/viz/kneel/kneeling_stop_003__A097.png) | 起跪过程没有脚下小柱，头顶空间清空，当前为G。 |
| 57 | [ladder_climbing_down_start_over_R_001__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/ladder/ladder_climbing_down_start_over_R_001__A300.png) | 下梯动作只得到高处窄柱和局部顶棚，缺少梯档与梯架，X排除合理。 |
| 58 | [ladder_jump_on_R_005__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/ladder/ladder_jump_on_R_005__A300.png) | 跳上梯子的动作没有攀附结构，只有地面，保持X排除。 |
| 59 | [ladder_slide_down_R_002__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/ladder/ladder_slide_down_R_002__A300.png) | 垂直下滑轨迹没有梯档或扶手，未用零碎平台补造梯子，保持X。 |
| 60 | [ladder_slide_down_R_004__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/ladder/ladder_slide_down_R_004__A300.png) | 高处下滑到站立只配地面，缺少连续攀附几何，保持X。 |
| 61 | [ladder_slip_R_002__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/ladder/ladder_slip_R_002__A300.png) | 梯上滑落动作没有梯架实体，场景仅保留地面，保持X。 |
| 62 | [lift_crate_come_up_50cm_box_R_005__A249](../../examples/results/bones_terrain_fit_v9_20261024/viz/known_dimension_support/lift_crate_come_up_50cm_box_R_005__A249.png) | 上箱支撑存在，但搬物前伸手势附近配了一整面墙，撑墙解释可疑。 |
| 63 | [mohak_backward_stop_002__A031](../../examples/results/bones_terrain_fit_v9_20261024/viz/quadruped/mohak_backward_stop_002__A031.png) | 手膝后退停止位于连续地面与顶棚之间，脚或膝下没有小柱。 |
| 64 | [mohak_loop_002__A031](../../examples/results/bones_terrain_fit_v9_20261024/viz/quadruped/mohak_loop_002__A031.png) | 长段手膝前进有连续顶棚，整条路径没有离散足部或膝部支撑箱。 |
| 65 | [mohak_start_001__A031](../../examples/results/bones_terrain_fit_v9_20261024/viz/quadruped/mohak_start_001__A031.png) | 手膝起步沿连续地面展开，顶棚覆盖活动范围，身体下无碎块。 |
| 66 | [on_all_fours_turn_270_R_003__A202](../../examples/results/bones_terrain_fit_v9_20261024/viz/quadruped/on_all_fours_turn_270_R_003__A202.png) | 手膝原地转向下方无小箱，顶棚覆盖转身区域。 |
| 67 | [on_all_fours_turn_360_R_002__A201](../../examples/results/bones_terrain_fit_v9_20261024/viz/quadruped/on_all_fours_turn_360_R_002__A201.png) | 手膝整圈转向保留地面与低顶棚，脚、膝下没有额外支撑。 |
| 68 | [stairs_climbing_down_loop_R_102__A301](../../examples/results/bones_terrain_fit_v9_20261024/viz/stairs/stairs_climbing_down_loop_R_102__A301.png) | 下楼动作只保留五个分离支撑，部分高度缺台阶且箱间有空隙，当前为R。 |
| 69 | [stairs_climbing_jog_down_loop_R_001__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/stairs/stairs_climbing_jog_down_loop_R_001__A300.png) | 跑下楼的高度持续下降，却没有任何台阶，仍被归E，存在明确漏建。 |
| 70 | [stairs_climbing_jog_down_stop_R_003__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/stairs/stairs_climbing_jog_down_stop_R_003__A300.png) | 下降路径只在末端留一个低台阶，前段台阶缺失，当前仍为E。 |
| 71 | [stairs_slip_R_003__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/stairs/stairs_slip_R_003__A300.png) | 七个支撑沿滑步路径排布，但间距和高度不连续，E分类仍保留这种稀疏楼梯。 |
| 72 | [stairs_slip_R_004__A300](../../examples/results/bones_terrain_fit_v9_20261024/viz/stairs/stairs_slip_R_004__A300.png) | 七个支撑构成离散高度序列，中间仍缺踏点，记录缩减证据和缺支撑拒绝。 |
| 73 | [turn_crouch_360_R_001__A194](../../examples/results/bones_terrain_fit_v9_20261024/viz/crouch/turn_crouch_360_R_001__A194.png) | 蹲姿原地转向由顶棚覆盖，下方只有连续地面，没有脚下小箱。 |
| 74 | [walk_backward_start_001__A029](../../examples/results/bones_terrain_fit_v9_20261024/viz/flat_control/walk_backward_start_001__A029.png) | 平地后退起步没有额外支撑块、侧墙或顶棚。 |
| 75 | [wall_leaning_idle_360_R_003__A285](../../examples/results/bones_terrain_fit_v9_20261024/viz/wall_contact/wall_leaning_idle_360_R_003__A285.png) | 靠墙静止姿态没有拟合墙面，图中只有地面和人体，当前仍归E。 |

## 8. 发布与复核

- [结果入口](../../examples/results/bones_terrain_fit_v9_20261024/README.md)：6,788行紧凑TSV、分组JSON、report、完整对照和测试日志。
- 发布 **75张PNG，10,967,917 bytes**，以及 **75条viewer JSON**和对应[index](../../examples/results/bones_terrain_fit_v9_20261024/viewer/index.json)；最大的viewer为142,425 bytes。
- 完整拟合在 `${TAK_ROOT}/runs/bones_fit_v9_full/`；缓存为 `runs/bones_motion_cache_s2/`，执行日志、JUnit、完成标记和审阅补图为 `runs/bones_fit_v9_full_coord/`。
- 本次传回的压缩包为11,514,303 bytes，最大单文件小于50MB，使用现有连接传回；未增加Drive迁移。
- 发布前运行 `tools/check_repository.py`，核对UTF-8/U+FFFD、TSV列宽、75张图的审查覆盖、viewer索引和80条上限。Controller完整失败记录随结果保留。
