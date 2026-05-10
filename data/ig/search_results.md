## IG search results (2026-05-10T06:49Z)
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/runner/work/qqq_swing_backtest/qqq_swing_backtest/src/ig_fetch.py", line 230, in <module>
    main()
  File "/home/runner/work/qqq_swing_backtest/qqq_swing_backtest/src/ig_fetch.py", line 220, in main
    search()
  File "/home/runner/work/qqq_swing_backtest/qqq_swing_backtest/src/ig_fetch.py", line 74, in search
    client.login()
  File "/home/runner/work/qqq_swing_backtest/qqq_swing_backtest/src/ig_client.py", line 233, in login
    raise IGError(f"login failed: HTTP {resp.status_code}: {snippet}")
src.ig_client.IGError: login failed: HTTP 403: {"errorCode":"error.security.api-key-invalid"}
```
