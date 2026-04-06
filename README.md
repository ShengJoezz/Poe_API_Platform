# Poe GPT-5.4-Pro 本地审计型客户端

这是一个本地运行的 GUI，但重点已经不只是“能聊天”。

这一版优先解决 Poe 使用里最危险的问题：

**结果没在本地可靠保存，但点数已经真实扣掉。**

所以它现在的核心目标是：

- 流式输出边收边写盘
- 每条请求都有本地唯一 ID
- 余额前后自动快照
- Usage API 自动同步与对账
- 请求状态机不再由界面观感决定
- 未完成请求可以恢复查看
- 一键导出证据包

## 现在已经对上的 Poe 核心能力

这版直接对齐了 Poe 官方提供的几组能力：

- [Responses API](https://creator.poe.com/docs/external-applications/responses-api)
- [OpenAI Compatible API](https://creator.poe.com/docs/external-applications/openai-compatible-api)
- [Get current point balance](https://creator.poe.com/api-reference/getCurrentBalance)
- [Get points history](https://creator.poe.com/api-reference/getPointsHistory)
- [Rate limits](https://creator.poe.com/api-reference/rate-limits)

## 这版新增的关键能力

### 1. 本地账本

所有请求都会记到本地 SQLite：

- 会话 ID
- 本地 request ID
- 模型
- 传输方式
- 请求状态
- 对账状态
- 输入摘要
- 附件清单
- 流式输出文本
- 前后余额
- Usage API 匹配结果
- 请求事件流

数据库位置：

```text
data/poe_audit.db
```

流式输出文件位置：

```text
data/streams/<request_id>.txt
```

### 2. 流式边收边存

这版不再把输出只放在前端内存里。

- 每收到一个 delta，就会立即 append 到本地文本文件
- 同时更新 SQLite 中的 `assistant_text`
- 即使页面断开，服务端也会继续把结果落到本地

### 3. 请求状态机

每条请求至少会落到这些状态之一：

- `queued`
- `streaming`
- `partial_saved`
- `completed`
- `timed_out`
- `failed`

另外还有独立的对账状态：

- `pending`
- `matched`
- `unmatched`

### 4. 余额与用量对账

每次请求都会：

1. 请求前自动抓一次 `current_balance`
2. 请求结束后再抓一次 `current_balance`
3. 自动同步 `points_history`
4. 把本地请求和 Usage API 条目按模型与时间窗口做匹配

这意味着以后你看到的不只是“可能扣了”，而是：

- 哪一条请求
- 对应哪个本地 request ID
- 余额变化多少
- Usage API 里的 `query_id`
- `cost_points` 是否和余额变化一致

### 5. 失败恢复与证据包

每条请求都支持导出证据包，里面会包含：

- `summary.md`
- `request.json`
- `transcript.json`
- `balance_snapshots.json`
- `request_events.json`
- `usage_entry.json`
- `stream_output.txt`

这部分通过界面里的“证据包”按钮触发。

## 当前的传输策略

### Responses API

用于纯文本轮次。

原因：

- 支持 `previous_response_id`
- 更适合减少重复发送整段历史
- 更贴近 Poe 官方现在推荐的 Responses 能力

### chat/completions

用于带附件的轮次。

原因：

- 这版附件走的是 Poe OpenAI 兼容接口的 `messages[].content`
- 支持 `text` / `image_url` / `file`

## 风险控制上已经做了什么

### 已实现

- 最小剩余点数保护
- 附件总大小限制
- 附件轮次默认只带最近若干轮历史，而不是无限堆上下文
- 没有自动重发整条高价请求
- 本地请求与 Poe Usage 分离记录，避免只信网页 UI

### 还没有做满的部分

下面这些方向已经留好了扩展空间，但这版还没有完全做成：

- 真正基于 `cost_points` 的请求前预估止损
- 自动摘要历史
- 长文档索引 / RAG
- 多会话搜索
- 更细粒度的恢复操作，比如“从 partial 输出继续”

## 测试模型说明

按 2026-04-03 查到的 Poe 官方页面，`Gemini-2.0-Flash-Lite` 仍然适合作为低成本测试模型。

所以当前 GUI 默认保留两套预设：

- `GPT-5.4-Pro`
- `Gemini-2.0-Flash-Lite`

## 启动方式

这一版已经改成：

- `Next.js` 负责前端
- `Python server.py` 继续负责 Poe 代理、本地账本、对账和证据包

推荐直接一起启动：

```bash
npm run dev:all
```

然后打开：

```text
http://127.0.0.1:3000
```

如果你要分开跑：

```bash
npm run backend
npm run dev
```

其中：

- 后端默认监听 `http://127.0.0.1:3030`
- 前端默认监听 `http://127.0.0.1:3000`

## 使用方式

1. 填入 Poe API Key
2. 先用 `Gemini-2.0-Flash-Lite` 验证链路
3. 需要高质量时再切回 `GPT-5.4-Pro`
4. 可选填写最小剩余点数保护
5. 可选限制附件上限 MB
6. 上传或拖拽附件
7. 发送消息
8. 在左侧账本面板里查看请求状态、余额、对账和证据包

## 目录

```text
.
├─ package.json
├─ server.py
├─ data/
│  ├─ poe_audit.db
│  └─ streams/
├─ public/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
└─ README.md
```

## 说明

- API Key 默认只存在页面内存里
- 勾选“在当前浏览器记住 API Key”后，会保存到浏览器本地 `localStorage`
- 服务端不会把 Key 写入磁盘
- 附件目前仍然是浏览器先转成 data URL 再发送，所以大文件会慢
- 这版优先做成单用户本地工具，不是多用户平台
- 如果后面要进一步产品化，可以继续拆成：
  - React / Next.js 前端
  - 更完整的会话搜索与筛选
  - 摘要/RAG
  - 更严格的止损策略
