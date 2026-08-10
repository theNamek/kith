# 掘金发布帖

**标题候选**(A 推荐):

- A: `我给 multi-agent 系统写了一层"社交记忆":谁靠谱、谁坑过你、凭什么这么说`
- B: `Agent 记忆都在卷事实检索,但真正让 multi-agent 翻车的是关系失忆`
- C: `开源 kith:让你的 agent 记住"谁是谁"——信任、可靠性、情绪,全部可溯源`

**标签**: AI Agent / LLM / 开源 / Python / 多智能体

---

## 正文

### 从一个真实的翻车现场说起

如果你跑过 multi-agent 系统,大概率见过这一幕:planner 把任务委托给某个
worker agent,失败;下一轮,**它又把任务委托给了同一个 worker**。第五次。

不是模型不够聪明——是系统里**没有任何地方记得前四次失败**。每个会话开始,
所有 agent 都是陌生人。

学术界对 multi-agent 失败模式做过系统分析(MAST 失败分类),排前面的
inter-agent misalignment(智能体间失配)、验证缺失,本质上都是**关系状态
的失忆**:信息一直都在,系统只是不记得"谁对谁做过什么"。

现有的 agent 记忆方案(Mem0 61k star、Zep、LangMem)都很优秀,但它们回答
的是同一个问题:**"我该记住什么事实?"** 没有人回答:

- 这条记忆是**关于谁**的?
- 它**允许谁看见**?(私聊里存的 API key 被注入到群聊——这不是检索 bug,
  是信任事故;hermes-agent 有 4 个 open issue 在吵这件事)
- 关系怎么**随时间演化**?(信任会衰减,坑过你两次的和十次交互前坑过你
  一次的,应该被区别记住)

所以我写了 **kith**(pip 包名 `kith-ai`,MIT,约 1000 行,stdlib+sqlite
零重依赖):**事实记忆记"发生了什么",kith 记"这件事对关系意味着什么"。**

### 三个原语

```python
import kith

store = kith.Store("sqlite:///team.db")
me = store.principal("agent:planner-7")

# Observation:append-only 的原始观察(交互/断言/情绪读数)
me.observe(subject="agent:coder-2", kind="interaction",
           payload={"promised": "fix by 5pm", "delivered": False},
           context="task:deploy-42")

# RelationshipView:派生的关系视图——信任、可靠性、情绪、能力
v = me.view("agent:coder-2")
if v.reliability < 0.4:
    plan.add_verification_step()   # 记住了,不再重蹈覆辙

# 每个数字都能自证:哪些观察、什么时候、为什么得出这个分
v.explain()
```

关键设计:**关系状态是算出来的,不是存出来的**。trust 分数带时间衰减
(30 天半衰期)、负面事件权重加倍(损失厌恶是信任研究里复现最多的发现
之一);嫌我的心理学模型太糙?derivers 全部是可插拔接口,换你自己的。
我们不主张心理学效度,主张**可审计性**。

### 泄漏路径测试:从一次被创始人 review 的教训说起

kith 的第二个卖点是 **Scope(可见性契约)**。这里有个故事:我之前给
hermes-agent(21.9 万 star)贡献 scoped memory,第一版只在系统提示注入
层做了过滤,创始人 Teknium review 时指出:错误响应会把整个存储作为
`current_entries` 返回——**换句话说,过滤了正门,密钥从错误消息的后门
泄漏了**。

这个教训在 kith 里升级成了第一设计原则:**访问边界是契约,不是过滤器**。
所有读路径(检索、导出、视图、错误消息、连 `repr` 都算)走同一个可见性
闸门,测试套件拿一个哨兵密钥字符串枚举每一个公开表面,包括两个容易被
忽略的通道:

- **派生值也是泄漏通道**:如果 Bob 对 Carol 的信任分被 Alice 的隐藏观察
  拉低了,分数本身就泄漏了隐藏数据的存在
- **存在性预言机**:查询一个"从没见过的人"和一个"有隐藏记录的人",返回
  的视图逐字节一致——你无法用 API 探测出"有人偷偷记了你什么"

### 可复现的效果(无 LLM 调用,带种子,秒级跑完)

**委托实验**:planner 给 20 个隐藏可靠性各异的 worker 派活,唯一变量是
有没有记忆:

```
policy                        failures  repeat-fail  retries
------------------------------------------------------------
baseline (无记忆)                100.6    83.1±32.9    100.6
kith (关系记忆)                   25.2    15.3±9.5      25.2

重复委托失败 ↓82%(50 workers 规模时 ↓79%)
```

**情绪传染实验**:16 个 agent 的小世界网络里埋一个"毒性源",每次交互
接收方记一条 affect observation。不接触任何人的内部状态,仅凭观察日志,
kith 的派生 sentiment 把毒性源定位在群体观感排名的最底端(5/5 种子正确)。
对做群体动力学研究的同学:**observation log 本身就是纵向数据集**。

### 三个 adapter,即插即用

- **LangGraph**:`observe_node` 包装任意节点,结果自动变成关系记忆;
  `KithSupervisor` 按 track record 选 worker(对着真实 StateGraph 集成
  测试,顺便替你踩了"节点名不能含冒号"的坑)
- **hermes-agent**:作为 MemoryProvider 插件运行,`on_delegation` 钩子
  让 subagent 委托结果零成本积累
- **A2A 协议**:终态 Task 自动映射成 outcome;AgentCard 的 skills 保持
  "自称"状态,直到真实任务证实——**名片吹的牛不算实绩**

### 一个贯穿始终的安全原则

模型永远不能自己决定"我是谁"。所有 principal 身份由运行时解析绑定,
模型面对的工具只有 `current`/`peer` 这样的解析令牌——它无法冒充别人
写观察,也无法猜一个 chat_id 去越权读取。(这条同样来自那次 hermes
review:第一版让模型自己构造 `platform:chat_id`,而它根本拿不到 ID。)

### 链接

- GitHub: https://github.com/theNamek/kith (设计文档、失败模式、非目标
  都在 docs/DESIGN.md)
- `pip install kith-ai`
- 两个 demo 都在 examples/,一条命令复现

我是最后一年博士生,研究方向是 multi-agent LLM 系统的群体情绪动力学,
kith 是研究的 systems 底座开源出来的版本。欢迎 issue/PR,也欢迎在评论
区拍砖——尤其是关于 trust 标量会不会诱导过度信任、scope 存成文本前缀
还是独立列这两个我自己也没完全想清楚的问题。
