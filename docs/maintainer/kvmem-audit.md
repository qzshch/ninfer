# KVMem 合并审计与推理验证

审计日期：2026-09-23。审计对象为 `feature/kvmem` 的 `7c93a004`，以上游共同祖先
`9e163eee` 为 NInfer 对照（50 个文件的改动），并参照本地 KVMem 实现 `13b7a15` 的
MeanK、query capture、Host/Device placement 语义。历史会话只用于寻找复现线索；下列结论来自
修复源码、独立算子 oracle 和实际模型执行。修复及回归工具随本审计记录一同提交。

## 结论与适用边界

2026-09-28 DFlash2 增量：全 local companion 的 KVMem + Vision + DFlash2 K7 已通过本机工程验证。
同一官方27B NVFP4产物，dense DFlash2 14/14、稀疏none 15/15、稀疏DFlash2 20/20、
36K×2 DFlash2+Vision 17/17；CTest6/6、Python合同48/48。每lane逻辑上限256K，
实际视觉输入各38923、输出各1024，87轮真实双lane decode；显存采样峰值31151MiB，
Windows commit94.58/95.92GiB、最小余量1.34GiB。覆盖真实零/部分/全部接受及草稿ring跨界。
当前huihui产物没有DFlash2权重，不与下方huihui/MTP结果混用。公共受控响应与none/dense对照一致，
但未测试两路实际满256K视觉或多轮统计质量。汇总：`out/dflash2-vision-20260928/qualification.json`。


2026-09-28 Vision 增量：补齐媒体 query 边界、整组 KV placement、GDN/Main/MTP replay、
可回放媒体 payload 生命周期，以及缓存媒体后的 MRoPE 文本后缀。当前用户 huihui 27B NVFP4
产物上，4K×2 MTP 13/13、none 15/15、dense MTP对照13/13、36K×2 MTP+Vision 14/14；
相关 CTest4/4，Linux工具合同45/45。36K组合配置各256K，实际视觉输入各38923token，
214轮真实双lane解码，显存采样峰值28873MiB、commit91.53/95.92GiB。没有测满两路256K视觉。
视频用例仅覆盖单帧经视频入口，未资格化真实多帧运动任务。详见
`out/vision-adaptation-20260928/qualification.json`；原始失败报告保留。本段为先前 none/MTP + Vision 资格化，DFlash2 增量单独列于下方。

原合并不能仅凭 HTTP 200 或一次长 prompt 成功判定可靠：检索 capture 没有接到实际执行路径，
而且缓存后的工具对话可触发 KV 恢复错误。已修复下表列出的执行、容量和生命周期问题，并提供
[可重复运行的测试管线](../../tests/e2e/README.md)。测试状态和具体配置以生成的报告为准。

当前代码范围是单 GPU、1–2 条 active lane、文本/视觉生成，支持普通解码、MTP 及全 local DFlash2 companion。
`--kvmem-window-pages 0` 保持密集模式。KVMem 模式支持 `--vision` 的媒体回放，明确拒绝第一代 DFlash、含 full-attention 层的 draft companion、scoring 和超过 2 条 lane。媒体按完整页组保留，最新可见媒体与 sink 为必留项；超窗口媒体在 admission 拒绝。文本单独请求沿用原选页规则。
每条 lane 独立持有 MeanK 索引、Q/K capture、已选历史页及 query replay 的 GDN 快照；DFlash2 另有独立 cyclic KV 快照（本机 companion 为每 lane 40 MiB）。
`--kvmem-window-pages` 是每条 lane 的窗口；`--kv-capacity auto` 为每条 lane 预留窗口、chunk
增长和原有 slack，上限为各自 logical context。Host KV arena 仍由所有请求共享。
2026-09-28 双 lane 增量验证：普通解码和 MTP 共 **16/16 E2E、9/9 CTest、42/42 Python 工具测试**。
配置为官方 Qwen3.8-27B NVFP4、INT8 KV、每 lane 4096-token 窗口、16K context、1024 chunk、
2 GiB 共享 Host KV。两种后端各两轮，两条 lane 均实际检索回 Host 历史页并进入同一解码批次；
8 份并发回复与各自独立执行的贪心文本完全一致。取消一条 lane 后，空槽在另一条仍生成时成功复用。
两组报告的二进制 SHA256 一致，汇总为 `out/kvmem-tests/dual-lane-qualification-20260928.json`。
这不覆盖所有窗口/上下文/模型组合，也不代表统计质量等价；历史单 lane 测试数字单独列于下方。
超过 Device window 的 prompt 从头计算；缓存尚未持有检索特征，因此不发布或恢复稀疏长上下文
continuation。窗口内仍可复用前缀，长请求后的小请求继续可用。

2026-09-23 单 lane 修正二进制的历史稳定性矩阵为 **95/95 E2E、9/9 CTest**，Python 工具合同 **41/41**。
其中完整 long profile 为 **30/30**；其余五组使用同一二进制的已通过报告。
原 long 报告的夹具未跨窗失败保留，修正夹具后重跑全部 long 检查，未修改失败标签。
证据汇总在 `out/kvmem-tests/service-projection-long-corrected-20260923/qualification.json`。
这不构成质量等价。原 1pp/95% 单一验收口径已被后续多轮配对 A/B 与 A/A 波动评估要求替代，
质量等价仍未验收。用户要求不自动运行完整 500 题；此前全量任务已停止。

当前实现使用 pre-RoPE block MeanK 与最后一条真实 user 消息中最多 512 token 的 Q 均值选择历史，
模板无法证明 token 边界时才回退至新 prefill 尾部。超窗请求在 query 前保存 GDN 状态，选页后
回退 Main/MTP KV 并重算 query 至 prompt 末尾，再发布首 token。预算分配、部分块特征、已接受
decode token 的特征累计和长前缀缓存所有权仍与参考实现有差异，尚未证明 KVMem 质量等价。
通过 needle fixture 不能证明 LongMemEval 等真实任务的质量等价，也不能证明任意模型/参数均稳定。
一次 194202-token needle 回复虽然最终代码正确，但前面有矛盾解释；这是已观察到的回答质量限制。
runner 要求最后一行答案正确，另用 `exact_answer_format` 记录是否只有答案，不把子串出现视为召回。

## 确认的问题和修复

| 严重性 | 原问题及影响 | 修复和证据 |
|---|---|---|
| P1 | query replay 原先只恢复 GDN，DFlash2 local cyclic KV 会保留 probe 状态；回放没有 draft feature sink。 | 保存/恢复每 lane 完整 local KV 与 query frontier，清空无效 pending features，回放时按原生 target 轴重新捕获 features，沿用 draft scalar logical positions。快照纳入 startup；拒绝尚未实现的 full draft KV。 |
| P1 | 草稿 prefill 控制只在请求开始时发布，decode/另一 lane 可以覆盖共享状态槽和 KV row。 | 每个 feature append 按当前 chunk 的实际 state slot/KV row 发布，涵盖普通 prefill、query replay 和 forced append。未新增数值内核。 |
| P1 | `TextContext::set_kvmem_capture` 没有调用方。Q/K capture 恒空，所谓检索实际上只剩结构选择/平分。 | 在 prefill 配置 execution card 时连接 capture；真实运行检查 Q norm 非零、历史块有分数且确实从 Host promoted。 |
| P1 | capture 固定 16 层、Q=6144、K=1024、8 slots；更换模型维度或 chunk 可能越界，raw `cudaMalloc` 也不属于启动容量计划。 | 从模型配置计算维度和 chunk slots，在 persistent arena 内规划。2048 chunk 的真模型回归覆盖扩展容量。 |
| P1 | 每个 chunk 只累计完整块，跨 chunk 的 128-token 块丢失；Q 平均整个 prompt，历史内容淹没查询。 | 全局 block ID 选择 ring slot，保留未完成块，仅清空已发布 slot；Q 限定尾部查询范围。 |
| P1 | capture/index 只在 root 路径清空，cache hit 或取消后的新请求可能继承另一请求的特征。 | 每次 admission 重置请求特征；缓存范围外从头 prefill。工具历史、断连和交替长短请求回归。 |
| P1 | prefill 提前 materialize `window + chunk`，随后按最大 mapped index 滚动，会让未来页挤掉尚需消费的历史。 | 只 materialize 下一 chunk，提交后再 demote 历史、增加 entitlement 并映射下一段。 |
| P1 | prefill 检索出的历史在第一次 decode 滚动时被纯 recency window 替换；MTP 选择也可能不同。 | decode 同时保留 retrieval 与滚动 recency，MTP 跟随相同历史集合。host helper 和跨页长输出回归。 |
| P1 | 小 prompt 也被稀疏选择，缓存的 mapped/working-set 不一致。基线工具对话返回 SSE error：`required=22 device=20 missing=0 mapped=22 working_set=20 reuse=1362`。 | 窗口内不稀疏化；长 prompt 不恢复缺特征的缓存；decode 跨窗后停止发布 continuation。原失败 fixture 在修复后通过。 |
| P1 | sparse reservation 只减 exclusive resident pages，忽略共享前缀占用；fork materialization 又使用完整逻辑 ceiling。 | 所有 resident pages 共同占据窗口预算；admission 与 materialization 使用相同物理需求。共享 4 页/物理 8 页测试修复前 `bad_alloc`、修复后通过。 |
| P1 | 请求准入不计后续 Host spill，可能接收永远无法完成的请求。 | 超窗请求计入 Main/MTP 全部 reserved context 与 staging 的 Host 峰值空间，由资源规划判定可行性。 |
| P1 | block-table 最终发布在 transfer stream，计算 stream 缺乏完成依赖。 | placement 返回前同步最终 table publication，避免计算读旧映射。 |
| P1 | 在 engine-wide 异常后无条件重置 `failed_`，可能继续使用损坏 CUDA context/不完整清理状态。 | 恢复 fail-stop；正常请求级错误和断连保留其原有局部处理路径。 |
| P2 | materialization audit 的固定 32 页宽限掩盖真实超额索取。 | 删除宽限；dense 恢复精确核对，sparse 的实际 Device claim 不能超过计划。 |
| P2 | inactive truncate 留下越界 working-set entries，后续 growth 产生过期/重复项；Host-only 页允许扩大 committed coverage。 | truncate 裁剪集合，新增 coverage 仍要求 Device 副本；存储生命周期回归。 |
| P2 | sparse 容量校验仍套用 dense 下限，auto 又可能超出小 context 的逻辑上限；不支持的组合没有拒绝。 | 容量以 window/chunk/margins 推导并受 logical ceiling 限制，增加启动组合校验及 options 测试。 |

用户已有的 `apps/serve/main.cpp` SIGPIPE 忽略改动保留。断开 SSE 客户端后，测试要求同一进程能
继续完成一次真实生成；单纯 `/health` 成功不足以通过。

## Query replay 质量修复（第二阶段）

实际参考 adapter 的 retrieval 路径使用按 layer/KV-head 平均的 MeanK cosine；不能从其他实验
接口或注释推断产品路径使用 attention softmax。新增的诊断集固定 400 条历史记录、seed 74191，
包括三个位置的单跳、双跳、事实更新、不可回答和 query 后长工具输出，共 15 题。完整回复必须是
精确的单字段 JSON；额外矛盾解释、重复字段、错误值和长度截断均失败。原始请求、答案和进程日志
保存在 `out/kvmem-tests/quality-query-20260923/`。

| 对照 | 正确数 | 结论 |
|---|---:|---|
| 第一阶段稳定性修复后、suffix query | 5/15 | 能运行不代表能召回 |
| 仅修 last-user query | 6/15 | 单改查询位置不足以恢复质量 |
| 同权重密集对照 | 15/15 | 题目在该权重上可答 |
| last-user + GDN/KV query replay | 14/15 | 相对旧路径提高 9 题 |
| 同实现开启 MTP | 14/15 | 与普通解码同一道题失败 |

稀疏配置均为 4096-token Device window、INT8 KV、32768 logical context、1024 prefill chunk，
实际 prompt 13181–16540 token，greedy、thinking 关闭。剩余 `two_hop-90` 应答
`KEY-39124558`，实际为 `NOT_FOUND`，未删除或重试掩盖。15 题属于缺陷定位，不能证明统计等价。

查询边界测试先复现了工具尾部和伪造 role 文本的定位错误；frontend provenance 修复后通过。
Host-only 后缀和 Both 部分尾页的 rewind 测试先在原 destructive truncate 路径失败，新增私有
replay 回退契约后通过；还检查回退后重新增长和全部资源释放。GDN 快照在启动 persistent arena
中规划，禁止在请求期额外分配未计账的 Device 内存。

参考版本固定 `13b7a155c6aaf038a02380b16bd8b8ae7eb760e7`、llama.cpp 子模块
`b81c99b479d4c24e5eeca10de99032ebd343ef8f` 加该版本累计补丁。现有 `.ninfer` 为 NVFP4/FP8
混合配方，参考 GGUF 为 NVFP4 backbone 加 BF16 embedding/head，不能声称权重表示一致。
跨引擎评分必须保留各自 dense 对照及量化、预算差异。参考 KVMem 的同组密集诊断已完成
15/15；尚无质量等价完成结论。

带 presence/frequency penalty 的复查暴露了额外问题：probe 的弃用首 token 会增加 sampler
计数。固定高惩罚配置的完整 JSON 评分为 8/15；直接状态回归在修复前记录
`KVMEM probe sample_history=1` 并失败。probe 现仅以无计数副作用的 argmax 生成临时 MTP
bridge token，正式采样留到回放，避免把未发布 token 写入生成历史。该修复的实机复验状态
为普通/MTP 高惩罚配置均 **14/15**，直接状态检查均为 `sample_history=0`，进程正常结束。
参考 KVMem 4096 选择预算加 1024 generation reserve 的稀疏诊断为 **11/15**：三个 tool-tail
和 two_hop-90 失败。其 mandatory query-tail 与本实现预算分配不同，不能据小样本宣称优于或等价。

长工具尾部 query replay 的取消缺陷已实测复现：下一请求等待 37.27 秒。修复将回放拆成
独立 cursor 驱动的单 chunk 步骤，每步同步后返回 Engine，prompt 进度不重复计数。
普通/MTP 的 135258-token 回放中断与后续生成均通过，下一请求分别为 0.0967/0.1035 秒。
当前完整 long 检查取消 180303-token 回放后，下次推理用时 0.0986 秒。

LongMemEval-S 完整 500 题已冻结，两引擎 32K pilot 与各自密集对照均完成十题预测，
全部零执行错误。新版 NInfer pilot 同样十题完成，但语义裁决尚未完成。
参考 pilot 曾分别触发 Linux OOM 和 Windows Commit 耗尽；失败记录完整保留。
Windows 事件 2004 记录 Commit 为 102730219520 / 102998466560 bytes（99.74%）。
恢复 WSL 24 GiB/8 GiB swap 后，参考采用 4 GiB Host arena 和进程级
`MALLOC_MMAP_THRESHOLD_=131072`，默认/关闭 CUDA Graphs 均完成十题且答案逐字相同，
匿名 RSS 峰值约 8.9 GiB、进程 swap 为零。支持分配器保留是增长的重要原因，
不能宣称任意长运行均无泄漏。runner 监测 Linux 余量，并保留触发保护的失败结果。
EvalScope SWE mini 的 50 题及两个 pilot 配置已冻结；真实 agentic 模型执行仍待完成。
两个标准补丁环境检查最初仅一题通过：Sphinx 三个 PASS_TO_PASS 测试因外部网络失败。
只给该次容器配置 NAS 现有代理后，两题标准补丁完整通过；未改镜像、测试或评分规则。
运行入口新增可记录的 `--container-proxy`，同时覆盖 agent 和独立评分容器；这不是模型成绩。

后续数据独立性检查发现 LongMemEval 有 29 组原题/abstention 变体及八组重复 evidence ID；
按两种关联的传递闭包固定成 466 组（432 单题、34 双题）。正式比较要求源 SHA256 绑定的分组，
采用保留 question-weighted accuracy 的组大小分层精确二项累计尾概率区间；仍需组间独立假设，
不能把全部 500 题无条件视为独立，未改变 1pp 门槛。

用户指定的 Qwen API 已可用。评分采用固定 source-grounded v3 rubric，完整答案及冻结的
answer-session 记录；这些评分证据不进入模型推理输入。最新校准 **11/12 符合预期、一题未决**。
参考 pilot 自动 base/strict 均为 8/10；NInfer 九题有效、一题因负面证据缺失未决，不报总分。
源审计确认鞋子答案日期写错，并发现公交费用题的 abstention gold 与保留的 assistant 历史
存在冲突，须对两引擎一致裁决；不能把旧 9/10 自动分数作为验收证据。
无效语义响应保留而不反复抽样；引用校验不证明裁判准确，模型别名也无不可变快照保证。
评分导入保留裁判配置、报告 hash、证据及裁决协议；比较拒绝不同裁决协议。
当前本地和 NAS Python 合同检查均 **40/40**，完整质量门槛尚未通过。

新增的分段回放曾漏计 service projection：`bounded-replay-long-20260923` 的 MTP
长输出触发 `request service projection consumed 1 quanta with 0 remaining`，后续
请求返回 503。root、candidate 和 sealed plan 现共用 query span 选择并计入回放量子，
包括 rewrite frontier 缩短的 chunk；首次探测 step 内的 checkpoint 切分不额外计数。
真实工具尾部 8075 tokens、`max_tokens=1` 的普通/MTP 回归各 **6/6** 通过，证明未借用
剩余输出预算；runtime-mechanisms 单测和 **39/39** Python 合同通过。新版二进制为
`5cf7af056cfa28a41c0ef92de25a1778627584b75743d0ad94fb5a35a990bca3`，
完整新版 long 检查 **30/30** 通过：两轮实际 245618-token prompt、768-token 输出、
跨窗 sampler history=0、one-token replay budget 和取消恢复全部通过。
结合该二进制未变的五组已通过报告为 **95/95 E2E、9/9 CTest**。
long 整卡显存采样峰值 30192 MiB，Windows Commit 峰值 80.23/95.92 GiB，
最小采样余量 15.70 GiB，未触发内存保护。新增检查进入常规回归与 long。

## 第一阶段数值与功能证据（query replay 修复前）

算子测试使用原测试框架的独立 FP64/量化 oracle，补充“sink + 尾部、中央有洞”的长 KV 输入，
让 prompt 路径和 small-T 路径真正访问 hole。覆盖 BF16、INT8、FP8、NVFP4、K8V4 五种 profile；
它检查掩码和 attention 数学，不依赖另一个 CUDA kernel 或模型输出作为 oracle。

存储测试覆盖 shared prefix 预算、sparse activation、Host/Device placement、truncate 后 growth、
释放后的物理资源归零。检索测试覆盖选择和 decode 保留历史页。Python runner 测试确保 HTTP 成功
后的 SSE error、缺少 `[DONE]`、空输出或缺少 usage 均会失败。

真模型为显式指定的官方 v3 `qwen3_8_27b_nvfp4.ninfer`（不是其他微调权重）。环境为
RTX 5090D 32 GiB、WSL Ubuntu 24.04、CUDA 13.1、sm_120a、Python 3.11。
模型 E2E 使用 INT8 KV，分别验证 MTP 和普通解码；其余 KV profile 的证据仅为算子 oracle。

最终管线依次执行构建、七项 CTest、runner 合同测试、JSON/SSE smoke、小窗口 MTP/普通解码回归、
dense smoke 和 256K 配置。长测采用 1536 页（96K）Device window、262144 logical context、
12 GiB Host KV；同一进程先形成工具历史缓存，再重复两轮 8K/32K/64K/96K/128K/192K/250K
nominal prompt，并检查实际 token usage。额外执行历史 needle 和至少 512 token 输出的多次页增长。

报告保存 `report.json`、JUnit、`server.log`、GPU 采样与源码元数据。任何失败不会自动重试，也不会
通过扩大 KV audit 宽限转成通过。运行器仅终止自己创建的进程；端口被其他服务占用则失败。

## 第一阶段执行结果（query replay 修复前）

完整 `long` 管线退出码为 **0**：7/7 CTest、6/6 runner 合同测试、88/88 E2E 检查通过。
本地结果摘要为 `out/kvmem-tests/qualified-20260923/RESULTS.md`，同目录保存原始报告。

| 组别 | 通过 | 关键覆盖 |
|---|---:|---|
| smoke | 9/9 | JSON、SSE、工具历史、大输出预算、断连恢复、关闭和日志 |
| MTP regression | 19/19 | 2048 chunk、两轮跨窗、9104-token needle、768-token 输出 |
| ordinary regression | 19/19 | 普通解码的同类缓存、跨窗和 768-token 输出 |
| dense smoke | 8/8 | 关闭 KVMem 的密集 KV 对照 |
| Host capacity | 6/6 | 32 MiB Host 容量立即拒绝超预算请求，随后正常生成 |
| long | 27/27 | 两轮最大 245618-token prompt、194202-token needle、107266-token prompt 后输出 768 token |

长测整卡采样峰值为 30460 MiB（约 29.75 GiB），包含其他进程，只是采样而非精确瞬时峰值。
needle 最终代码正确，但 `exact_answer_format=false`；未把额外解释隐去或宣称完全遵守格式。
所有本次测试服务均已退出。

原始基线在 15 个检查中有 2 个失败（工具历史恢复及相应错误日志），修复中间版本的相同 15 项通过。
早期管线曾因同端口 TIME_WAIT 中断；已改用每配置独立端口，并保留失败记录于
`out/kvmem-tests/final-20260923/`，未将该次基础设施失败改写为成功。

## 操作入口

```bash
export PYTHON=/home/druid/.local/bin/python3.11
export NINFER_MODEL='/mnt/d/LLM Model/qwen3_8_27b_nvfp4.ninfer'
bash tests/e2e/run_kvmem_pipeline.sh smoke
bash tests/e2e/run_kvmem_pipeline.sh regression
bash tests/e2e/run_kvmem_pipeline.sh long
```

GitHub Actions 提供手动 `KVMem GPU regression` 工作流，需要配置 `ninfer-sm120a` 自托管 Linux
runner 和 `NINFER_E2E_MODEL` 仓库变量；Python 可用 `NINFER_E2E_PYTHON` 指定。工作流文件存在
不代表远端 CI 已配置或运行。本次本地输出放在 `out/kvmem-tests/`，不随源码上传。
