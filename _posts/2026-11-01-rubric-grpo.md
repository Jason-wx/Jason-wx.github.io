---
layout: post
title: "当 Rubric 遇上 GRPO：被压扁的奖励信号"
subtitle: "关于方差、有效权重与信用分配：rubric 奖励在 GRPO 里到底教了模型什么"
---

{% raw %}
<p>我最近一直在用“信号压缩”的角度看 rubric 奖励。RLVR 的成功很大程度上来自一个朴素的配方：答对给 1、答错给 0，再用 GRPO 把答对的概率推高。奖励足够简单、足够干净，这是它在数学和代码上效果惊人的原因。</p>
<p>但大多数真实任务没有标准答案。于是社区开始用 rubric 描述“什么是好回答”：把质量拆成一条条可判定的标准，由 LLM 逐条打分，再汇总成奖励。这看起来只是把“一个判定”换成了“k 个判定”。我的观点是：GRPO 的很多设计都默认奖励是干净的二元信号，换成 rubric 之后，信号会在三个地方被压扁，而每次压扁都会悄悄改变模型实际在学什么。</p>
<p>这不是严格的推导，我也没有做系统的实验。文中的数字例子都是构造的 toy example，用来说明机制；涉及实证的结论引用自已有工作。我只是觉得这个心智模型能解释不少 rubric RL 里看到的现象，值得写下来。</p>

<h2>GRPO 是为二元奖励设计的</h2>
<p>GRPO 的核心想法是去掉价值网络：对同一道题采样 G 个回答，让它们互为基线。每个回答的 advantage 由组内统计量归一化得到，然后这个数值被原封不动地赋给回答里的每一个 token。</p>
<div class="math">
$$
A_i = \frac{R_i - \mathrm{mean}(R_1,\dots,R_G)}{\mathrm{std}(R_1,\dots,R_G)}
$$
</div>
<p>这个式子里藏着三个设计选择：</p>
<ol>
<li><strong>减去组均值</strong>：只比较组内谁好谁坏，题目的绝对难度不重要。</li>
<li><strong>除以组内标准差</strong>：每道题的梯度量级被拉平，难题易题影响力相近。</li>
<li><strong>序列级广播</strong>：同一个回答的所有 token 共享同一个 advantage。</li>
</ol>
<figure><svg viewBox="0 0 760 232" role="img" aria-label="GRPO 的三个设计都默认奖励是干净的二元信号" font-size="13"><defs><marker id="grpo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--cds-chart-axis)"></path></marker></defs><text data-claude-text-id="title" x="24" y="34" font-size="15" font-weight="600" fill="var(--cds-text-primary)">GRPO 的三个设计都默认奖励是干净的二元信号</text><g data-claude-anchor="connectors" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"><path d="M148 96H168" marker-end="url(#grpo-arrow)"></path><path d="M292 96H312" marker-end="url(#grpo-arrow)"></path><path d="M436 96H456" marker-end="url(#grpo-arrow)"></path><path d="M580 96H600" marker-end="url(#grpo-arrow)"></path><path d="M374 124V160" stroke-dasharray="4 4"></path><path d="M518 124V160" stroke-dasharray="4 4"></path><path d="M662 124V160" stroke-dasharray="4 4"></path></g><g data-claude-anchor="step-sample"><rect x="24" y="68" width="124" height="56" rx="8" fill="none" fill-opacity="1" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="s1-name" x="86" y="92" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">采样回答</text><text data-claude-text-id="s1-sub" x="86" y="111" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">同一题 G 次</text></g><g data-claude-anchor="step-score"><rect x="168" y="68" width="124" height="56" rx="8" fill="none" fill-opacity="1" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="s2-name" x="230" y="92" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">逐个打分</text><text data-claude-text-id="s2-sub" x="230" y="111" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">对记 1，错记 0</text></g><g data-claude-anchor="step-mean"><rect x="312" y="68" width="124" height="56" rx="8" fill="var(--cds-chart-categorical-1)" fill-opacity="0.12" stroke="var(--cds-chart-categorical-1)" stroke-width="2"></rect><text data-claude-text-id="s3-name" x="374" y="92" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">① 减组均值</text><text data-claude-text-id="s3-sub" x="374" y="111" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">只比组内高低</text></g><g data-claude-anchor="step-std"><rect x="456" y="68" width="124" height="56" rx="8" fill="var(--cds-chart-categorical-1)" fill-opacity="0.12" stroke="var(--cds-chart-categorical-1)" stroke-width="2"></rect><text data-claude-text-id="s4-name" x="518" y="92" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">② 除组内 std</text><text data-claude-text-id="s4-sub" x="518" y="111" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">每题梯度等量</text></g><g data-claude-anchor="step-broadcast"><rect x="600" y="68" width="124" height="56" rx="8" fill="var(--cds-chart-categorical-1)" fill-opacity="0.12" stroke="var(--cds-chart-categorical-1)" stroke-width="2"></rect><text data-claude-text-id="s5-name" x="662" y="92" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">③ 广播到 token</text><text data-claude-text-id="s5-sub" x="662" y="111" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">全序列共享 A</text></g><g data-claude-anchor="assumption"><rect x="312" y="160" width="412" height="44" rx="8" fill="var(--cds-chart-reference-tint)" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="assumption-words" x="518" y="187" text-anchor="middle" fill="var(--cds-text-primary)">这三步都默认：奖励是干净的二元信号</text></g></svg><figcaption>GRPO 的流程。高亮的三步各自隐含一个前提：减组均值假设组内高低有意义；除以 std 假设方差反映模型的不确定性；序列级广播假设结果好坏与过程质量强相关。二元奖励下这三个前提基本成立，后文的问题都出在 rubric 打破了它们。</figcaption></figure>
<p>在二元奖励下，这三点都成立：</p>
<ul>
<li>组内标准差为 sqrt(p(1−p))，只要有对有错就有下界（G=16 时约 0.24），除以它不会放大假信号；</li>
<li>全对或全错的组没有梯度，可以直接丢掉（DAPO 的 dynamic sampling）；</li>
<li>数学推理步步相依，答案对错与过程质量高度相关，把结果奖励分给每个 token 是合理近似。</li>
</ul>
<p>换句话说，GRPO 能这么简洁，是因为它站在“奖励是干净的二元信号”这个前提上。后面要讲的所有问题，本质上都是这个前提不再成立。</p>
<p>还有一点容易被忽略：GRPO 的很多“默认设置”也是在这个前提下调出来的。去掉 KL 惩罚、放宽裁剪范围、按 token 平均 loss，这些做法在 RLVR 里被反复验证有效，原因之一是奖励本身很少说谎：模型想拿到 1 分，只能真的把题做对。当我们把同一套配置原封不动地搬到 rubric 奖励上时，等于默认 judge 也同样不会说谎。这个默认，正是后面所有问题的起点。</p>

<h2>从数学到 Agent，奖励越来越难写</h2>
<p>三类推理任务的奖励，写起来一个比一个难：</p>
<ol>
<li><strong>数学</strong>：解析最终答案，与标准答案比对，天然二元。</li>
<li><strong>代码</strong>：单元测试全过记 1，或用通过率做连续奖励。通过率其实已是“迷你 rubric”，每个测试一个条目、等权平均，也会出现过一两个简单测试就拿分的情况。</li>
<li><strong>Agent</strong>：<span class="hl">只能在终点检查（隐藏测试、环境终态）。轨迹长、奖励极稀疏，难题上大量组全零；过程中的危险或低效操作，终态也看不到</span>。</li>
</ol>
<p>Rubric 就是在这个背景下进入推理 RL 的。它为每道题写一组专属标准，每条带一个权重，可以是正的（Essential、Important、Optional），也可以是负的（Pitfall，即常见错误）。LLM judge 逐条判断是否满足，再加权汇总：</p>
<div class="math">
$$
r(x, y) = \frac{\sum_{j=1}^{k} w_j\, c_j(x, y)}{\sum_{j=1}^{k} w_j}, \qquad c_j \in \{0, 1\}
$$
</div>
<p>令 k=1、c1、判定函数为答案匹配，这个式子就退化成 RLVR。所以 rubric 常被看作 RLVR 的自然推广。HealthBench 用医生撰写的近五万条标准做评测，Scale AI 的 RaR 首次把实例级 rubric 用作在线 RL 奖励，之后 RLCF、Rubicon、OpenRubrics、Kimi K2 等工作把它扩展到了指令遵循、开放写作和工业级训练。</p>
<p>为了让后面的讨论更具体，先看一份 rubric 大概长什么样。RaR 论文里有一道医学题：一位 65kg 的患者出现严重代谢性酸中毒，问前 4 小时应补多少碳酸氢钠。它的 rubric 有七条左右，大致是这样分层的：</p>
<ul>
<li><strong>Essential</strong>：用对“碱缺失 × 体重 × 0.3”的公式；给出前 4 小时约 150 mEq 的明确建议；</li>
<li><strong>Important</strong>：解释为什么一开始只做部分纠正；写出完整的计算步骤；准确引用患者的体重和血气数值；</li>
<li><strong>Optional</strong>：顺带提一句碱缺失 −40 意味着酸中毒很严重；</li>
<li><strong>Pitfall</strong>：没有提到快速过度纠正的风险。</li>
</ul>
<p>这份 rubric 看起来很合理：它不只检查答案对不对，还检查推理过程、安全意识和表达完整性。问题在于，<span class="hl">当这些条目被丢进 GRPO 之后，它们各自对训练的影响力，并不等于我们写下的权重</span>。接下来的三节，都是在解释这句话。</p>

<h2>一个心智模型：三次压缩</h2>
<p>从 rubric 的判定结果到最终的参数更新，信号要经过三步：k 个条目的 0/1 判定先被加权聚合成一个标量，再在组内归一化成 advantage，最后被广播给回答里的每一个 token。每一步都丢掉了一些信息，也各自引入了一种偏差。</p>
<figure><svg viewBox="0 0 760 280" role="img" aria-label="三次压缩各引入一种偏差" font-size="13"><defs><marker id="comp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--cds-chart-axis)"></path></marker></defs><text data-claude-text-id="title" x="24" y="34" font-size="15" font-weight="600" fill="var(--cds-text-primary)">三次压缩各引入一种偏差</text><g data-claude-anchor="connectors" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"><path d="M178 100H214" marker-end="url(#comp-arrow)"></path><path d="M362 100H398" marker-end="url(#comp-arrow)"></path><path d="M546 100H582" marker-end="url(#comp-arrow)"></path><path d="M288 128V184" stroke-dasharray="4 4"></path><path d="M472 128V184" stroke-dasharray="4 4"></path><path d="M656 128V184" stroke-dasharray="4 4"></path></g><g data-claude-anchor="stage-vector"><rect x="30" y="72" width="148" height="56" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="vec-name" x="104" y="96" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">条目判定向量</text><text data-claude-text-id="vec-sub" x="104" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">k 个 0/1 判定</text></g><g data-claude-anchor="stage-aggregate"><rect x="214" y="72" width="148" height="56" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="agg-name" x="288" y="96" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">聚合为标量</text><text data-claude-text-id="agg-sub" x="288" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">r = Σ w·c / W</text></g><g data-claude-anchor="stage-normalize"><rect x="398" y="72" width="148" height="56" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="norm-name" x="472" y="96" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">组内归一化</text><text data-claude-text-id="norm-sub" x="472" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">A = (r − μ) / σ</text></g><g data-claude-anchor="stage-broadcast"><rect x="582" y="72" width="148" height="56" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="bc-name" x="656" y="96" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">广播给每个 token</text><text data-claude-text-id="bc-sub" x="656" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">所有 token 共享 A</text></g><g data-claude-anchor="bias-item"><rect x="204" y="184" width="168" height="72" rx="8" fill="var(--cds-chart-status-warning)" fill-opacity="0.12" stroke="var(--cds-chart-status-warning)" stroke-width="1.25"></rect><text data-claude-text-id="b2-name" x="288" y="208" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">条目层失真</text><text data-claude-text-id="b2-l1" x="288" y="228" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">有效权重 ≠ 设计权重</text><text data-claude-text-id="b2-l2" x="288" y="244" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">梯度流向提及类条目</text></g><g data-claude-anchor="bias-prompt"><rect x="388" y="184" width="168" height="72" rx="8" fill="var(--cds-chart-status-warning)" fill-opacity="0.12" stroke="var(--cds-chart-status-warning)" stroke-width="1.25"></rect><text data-claude-text-id="b3-name" x="472" y="208" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">题目层失真</text><text data-claude-text-id="b3-l1" x="472" y="228" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">样本权重由方差决定</text><text data-claude-text-id="b3-l2" x="472" y="244" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">小方差组放大噪声</text></g><g data-claude-anchor="bias-token"><rect x="572" y="184" width="168" height="72" rx="8" fill="var(--cds-chart-status-warning)" fill-opacity="0.12" stroke="var(--cds-chart-status-warning)" stroke-width="1.25"></rect><text data-claude-text-id="b4-name" x="656" y="208" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">Token 层失真</text><text data-claude-text-id="b4-l1" x="656" y="228" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">局部错误被一起强化</text><text data-claude-text-id="b4-l2" x="656" y="244" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">补偿聚合与伪相关</text></g></svg><figcaption>Rubric 奖励进入 GRPO 要经过三次压缩。上排是信号的流向：k 个条目的 0/1 判定先被加权聚合成标量，再在组内归一化，最后广播给每个 token。下排是每一步丢失信息后带来的偏差。</figcaption></figure>
<p>用一个类比来理解这三次压缩：一位老师拿着评分细则批改作文。</p>
<p>她本来逐条记下了“论点清晰 ✓、论据充分 ✓、有一处史实错误 ✗、结尾仓促 ✗”，但发回给学生的只有一个总分 82（<strong>第一次压缩</strong>）。接着，她不看绝对分数，只看这篇在同一批作文里排第几，并且规定每一批的分差都要拉到一样大，哪怕这批作文其实都差不多（<strong>第二次压缩</strong>）。最后，学生拿到“比平均高一点”的反馈，于是把这篇作文里的每一句话都当成值得保留的写法，包括那处史实错误（<strong>第三次压缩</strong>）。</p>
<p>每一步单独看都说得过去，但学生最终学到的东西，和细则想教的东西之间已经隔了好几层。</p>

<h2>Rubric 在悄悄决定哪道题更重要</h2>
<p>在 GRPO 里，一道题对训练有多大影响，完全取决于它的组内奖励方差。二元奖励下，这个方差只反映一件事：模型在这道题上答得稳不稳。但换成 rubric 之后，方差里混进了很多和“模型答得怎样”无关的东西。</p>
<p>把 rubric 分数的方差展开，可以看得很清楚：</p>
<div class="math">
$$
\mathrm{Var}(r) = \frac{1}{W^2}\Big[\sum_j w_j^2\, p_j(1-p_j) + \sum_{j\ne l} w_j w_l\, \mathrm{Cov}(c_j, c_l)\Big] + \sigma^2_{\text{judge}}
$$
</div>
<p>这里 p_j 是当前策略满足第 j 条的概率。方差至少受五样东西影响：</p>
<ol>
<li><strong>条目数量</strong>：条目越多，方差约按 1/k 缩小，奖励越平滑。</li>
<li><strong>权重集中度</strong>：一条权重独大时，rubric 退化成近似二元。</li>
<li><strong>条目相关性</strong>：几条标准查同一件事，会全中或全不中，既放大方差，又让这个方面被重复计分。</li>
<li><strong>条目相对当前模型的难度</strong>：已饱和或完全够不着的条目，p_j(1−p_j)=0，不贡献方差。</li>
<li><strong>judge 噪声</strong>：与回答质量无关。</li>
</ol>
<p>前三项在设计 rubric 时就定了，第四项随训练漂移，第五项是纯噪声。</p>
<p>把这放回刚才那道碳酸氢钠的题上想一想。训练到中途，模型大概已经稳定地用对公式、给出正确剂量，两条 Essential 几乎每次都满足；剩下的差异集中在“有没有解释部分纠正的原因”“有没有提到严重程度”这类条目上。于是同一组 16 个回答的分数挤在 0.8 到 0.9 之间，组内方差很小。</p>
<p>而另一道题，可能是一个开放式的咨询问题，rubric 只有四五条，其中一条“是否建议立即就医”权重很大、模型时对时错。它的组内分数会在 0.3 和 0.9 之间来回跳，方差很大。</p>
<p>那么，这两道题在 GRPO 眼里有什么区别？</p>
<p>这带来两难。先看原版 GRPO（除以标准差）如何处理下面两道题：</p>
<figure><svg viewBox="0 0 760 290" role="img" aria-label="差异悬殊的两道题，归一化后拿到同等梯度" font-size="13"><defs><marker id="var-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--cds-chart-axis)"></path></marker></defs><text data-claude-text-id="title" x="24" y="34" font-size="15" font-weight="600" fill="var(--cds-text-primary)">差异悬殊的两道题，归一化后拿到同等梯度</text><g data-claude-anchor="connectors" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"><path d="M260 100H298" marker-end="url(#var-arrow)"></path><path d="M446 100H496" marker-end="url(#var-arrow)"></path><path d="M260 196H298" marker-end="url(#var-arrow)"></path><path d="M446 196H496" marker-end="url(#var-arrow)"></path></g><g data-claude-anchor="q-a-rewards"><rect x="24" y="70" width="236" height="60" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="qa-name" x="142" y="95" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">题 A：奖励挤在一起</text><text data-claude-text-id="qa-vals" x="142" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">0.86 · 0.88 · 0.87 · 0.89</text></g><g data-claude-anchor="q-a-std"><rect x="298" y="70" width="148" height="60" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="qa-std" x="372" y="95" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">std ≈ 0.011</text><text data-claude-text-id="qa-std-sub" x="372" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">组内差异很小</text></g><g data-claude-anchor="q-a-adv"><rect x="496" y="70" width="240" height="60" rx="8" fill="var(--cds-chart-status-warning)" fill-opacity="0.12" stroke="var(--cds-chart-status-warning)" stroke-width="1.5"></rect><text data-claude-text-id="qa-adv" x="616" y="95" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">advantage 最大约 ±1.3</text><text data-claude-text-id="qa-adv-sub" x="616" y="115" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">差异可能只是 judge 抖动</text></g><g data-claude-anchor="q-b-rewards"><rect x="24" y="166" width="236" height="60" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="qb-name" x="142" y="191" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">题 B：奖励拉得很开</text><text data-claude-text-id="qb-vals" x="142" y="211" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">0.20 · 0.90 · 0.30 · 0.85</text></g><g data-claude-anchor="q-b-std"><rect x="298" y="166" width="148" height="60" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="qb-std" x="372" y="191" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">std ≈ 0.31</text><text data-claude-text-id="qb-std-sub" x="372" y="211" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">组内差异明显</text></g><g data-claude-anchor="q-b-adv"><rect x="496" y="166" width="240" height="60" rx="8" fill="var(--cds-chart-status-good)" fill-opacity="0.12" stroke="var(--cds-chart-status-good)" stroke-width="1.5"></rect><text data-claude-text-id="qb-adv" x="616" y="191" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">advantage 最大约 ±1.2</text><text data-claude-text-id="qb-adv-sub" x="616" y="211" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">反映真实质量差异</text></g><text data-claude-text-id="col-1" x="142" y="60" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">组内 4 个回答的奖励</text><text data-claude-text-id="col-2" x="372" y="60" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">组内标准差</text><text data-claude-text-id="col-3" x="616" y="60" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">除以 std 之后</text><text data-claude-text-id="note" x="24" y="262" font-size="11.5" fill="var(--cds-text-secondary)">归一化后两题梯度量级相同，但题 A 的“信号”可能主要是噪声</text></svg><figcaption>一个构造的 toy example。题 A 的四个回答几乎一样好，奖励只差 0.03；题 B 的回答有明显的好坏之分。除以组内标准差后，两题的 advantage 量级几乎相同，但题 A 的“差异”很可能只来自 judge 在某个可选条目上的抖动。</figcaption></figure>
<p>题 A 那 0.03 的差别，很可能只是 judge 在某个可选条目上翻转了一次，模型在认真拟合噪声。二元奖励没有这个问题：标准差有下界，零方差组可直接过滤；rubric 分数连续，标准差几乎不为零却可以任意小，DAPO 式过滤基本失效。</p>
<p>有意思的是，同一个机制在 RLVR 里通常被看作好事。组内方差大，说明模型在这道题上拿不准，除以 std 会让更新变小；方差小、模型稳定答对，更新反而更激进。<a href="https://arxiv.org/abs/2507.05386">Lai et al., 2025</a> 把这理解为 RL 自带的一种数据相关正则。</p>
<p><span class="hl">但在 rubric 场景里，这个解读反过来了：<strong>方差小不再代表“模型很确定”，而可能代表“rubric 已经分不出好坏”。</strong> 这时放大更新，放大的就是噪声。</span></p>
<p>反过来不除标准差（Dr. GRPO 的做法），影响力正比于方差：<span class="hl">条目少、权重集中的粗糙 rubric 话语权更大，设计精细的反而被冷落</span>。两种做法都不对——真正该决定题目权重的是<strong>信噪比</strong>，而 GRPO 对此一无所知。</p>
<p>至于天然偏爱高分或低分的 rubric：</p>
<ul>
<li>纯 GRPO 会减去组均值，绝对水平不直接影响梯度；</li>
<li>但奖励在 [0,1] 内时方差不超过 μ(1−μ)，均值靠近两端就意味着方差小，又回到噪声放大；</li>
<li>若用全局基线，或把二元 RLVR 奖励与连续 rubric 奖励混在一个 batch，绝对水平会直接改变 advantage 的正负，让某些领域被系统性偏爱。</li>
</ul>

<h2>梯度流向最容易刷分的条目</h2>
<p>上一节讲的是“哪道题被重视”，这一节往下看一层：在一道题内部，哪些条目真正在驱动梯度？</p>
<p>把 advantage 按条目展开，第 j 条对梯度的贡献大致正比于它的权重乘以它在当前组内的区分度：</p>
<div class="math">
$$
\text{有效权重}_j \;\propto\; w_j \cdot \sqrt{p_j(1-p_j)}
$$
</div>
<p>这意味着设计者写下的权重只是故事的一半。一条 Essential 标准，比如“给出正确诊断”，模型往往很早就学会了，p_j 接近 1，区分度趋近于零，它几乎不再产生梯度。而那些“是否提及某点”“是否给出额外例子”之类的 Optional 标准，满足率常年在 50% 上下，又恰好只要多写几句就能满足，于是吃掉了大部分梯度。</p>
<p>用一个简单的数字推演感受一下。假设一道题里有两条标准：</p>
<ul>
<li>“给出正确剂量”，Essential，权重 5；</li>
<li>“顺带提一下严重程度”，Optional，权重 2。</li>
</ul>
<p>训练到某个阶段，模型给出正确剂量的概率是 95%，提到严重程度的概率是 50%。代入有效权重的公式，前者约为 5 × √(0.95 × 0.05) ≈ 1.09，后者约为 2 × √(0.5 × 0.5) = 1.0。设计者认为重要 2.5 倍的条目，此刻对梯度的贡献和可选条目几乎一样。</p>
<p>再往后训练一点，正确剂量的概率到了 99%，它的有效权重掉到约 0.5，而可选条目仍然停在 1.0 附近——<strong>可选条目反过来成了主导。</strong> 这里的数字是我随手构造的，但趋势不依赖具体数值：<span class="hl"><strong>只要一条标准被学会了，它就会从梯度里“退场”，把位置让给那些还没被学会、而且往往更容易靠多写来满足的条目。</strong></span></p>
<p>关于 rubric RL 的 reward hacking，最常见的解释是 judge 不够强、被模型骗了。这当然是一部分原因，但我觉得不完整。<a href="https://arxiv.org/abs/2605.12474">Mahmoud et al., 2026</a> 的观察是：</p>
<ul>
<li>训练后完整性、提及类条目得分持续上升；</li>
<li>事实正确性、简洁性和整体质量反而下降；</li>
<li>弱验证器上的分数提升迁移不到更强的评判者。</li>
</ul>
<p>更关键的是，他们发现即使换成更强的验证器，只要 rubric 漏掉了重要的失败模式，hacking 依然存在：rubric 评判者偏好 RL 之后的模型，而不带 rubric 的整体评判反而偏好基座模型。</p>
<p>我更倾向的解释是方差结构。<span class="hl"><strong>梯度流向最容易移动的条目，而它们恰好靠堆砌内容就能满足。hacking 不只是 judge 被骗，更是方差结构把优化导向了错误的地方。</strong></span></p>
<p>这还是一个动态过程：</p>
<figure><svg viewBox="0 0 760 240" role="img" aria-label="梯度随训练从正确性流向提及类条目，最后流向噪声" font-size="13"><defs><marker id="phase-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--cds-chart-axis)"></path></marker></defs><text data-claude-text-id="title" x="24" y="34" font-size="15" font-weight="600" fill="var(--cds-text-primary)">梯度随训练从正确性流向提及类条目，最后流向噪声</text><g data-claude-anchor="connectors" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"><path d="M242 100H274" marker-end="url(#phase-arrow)"></path><path d="M486 100H518" marker-end="url(#phase-arrow)"></path></g><g data-claude-anchor="phase-early"><rect x="30" y="64" width="212" height="72" rx="8" fill="none" fill-opacity="1" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="p1-name" x="136" y="88" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">训练早期</text><text data-claude-text-id="p1-l1" x="136" y="108" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">Essential 满足率约一半</text><text data-claude-text-id="p1-l2" x="136" y="124" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">梯度主要用于学正确性</text></g><g data-claude-anchor="phase-mid"><rect x="274" y="64" width="212" height="72" rx="8" fill="var(--cds-chart-status-warning)" fill-opacity="0.12" stroke="var(--cds-chart-status-warning)" stroke-width="2"></rect><text data-claude-text-id="p2-name" x="380" y="88" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">训练中期</text><text data-claude-text-id="p2-l1" x="380" y="108" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">Essential 饱和，梯度归零</text><text data-claude-text-id="p2-l2" x="380" y="124" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">提及类条目吃掉大部分梯度</text></g><g data-claude-anchor="phase-late"><rect x="518" y="64" width="212" height="72" rx="8" fill="none" fill-opacity="1" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="p3-name" x="624" y="88" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">训练后期</text><text data-claude-text-id="p3-l1" x="624" y="108" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">大多数条目都已饱和</text><text data-claude-text-id="p3-l2" x="624" y="124" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">judge 噪声驱动更新</text></g><g data-claude-anchor="time-axis"><line x1="30" y1="176" x2="728" y2="176" stroke="var(--cds-chart-axis)" stroke-width="1.25" marker-end="url(#phase-arrow)"></line><text data-claude-text-id="axis-label" x="728" y="166" text-anchor="end" font-size="11.5" fill="var(--cds-text-secondary)">训练步数</text></g><text data-claude-text-id="formula" x="24" y="214" font-size="11.5" fill="var(--cds-text-secondary)">有效权重 ∝ w · √(p(1−p))：条目一旦饱和，贡献趋近于零</text></svg><figcaption>有效权重随训练迁移的示意，并非实测曲线。<span class="hl"><strong>早期 Essential 条目满足率在一半左右，区分度最高；中期它们饱和，梯度转向“是否提及”这类靠多写就能满足的条目；后期大多数条目饱和，剩下的组内差异主要来自 judge 噪声。</strong></span></figcaption></figure>
<p>没有人设计过这张课程表，它由方差结构自发产生。同样的逻辑也决定了条目冲突的走向：“简洁”与“完整”同权时，完整几乎总赢，因为多写比精准地删容易得多。</p>

<h2>序列级 advantage 看不见局部错误</h2>
<p>设想一个回答总分 0.8，advantage 为正，但其中有一句话犯了事实错误，违反了一条 Pitfall。GRPO 会把正的 advantage 分给所有 token，那句错话的概率也被一起推高了。</p>
<p>还是用那道碳酸氢钠的题。设想模型写出了一个相当不错的回答：公式用对了，计算步骤完整，解释了为什么只做部分纠正，最后还提醒要监测血气。唯一的问题是，它在解释部分纠正时顺手编了一个“临床指南推荐每小时不超过 50 mEq”的说法（这个数字是它自己发明的）。</p>
<p>这个回答满足了绝大多数条目，总分很高，在组内排名靠前，于是拿到一个明显的正 advantage。从 GRPO 的视角看，这个回答里的每一个 token 都“做对了”，包括那句编出来的指南。</p>
<p>先要澄清一点：<span class="hl"><strong>对于给定的标量奖励，策略梯度在期望上是无偏的</strong></span>。同一类错误会出现在很多样本里，有时在高分回答中，有时在低分回答中；如果它确实拉低奖励，平均下来它的概率会被压下去。所以“整体为正、局部应为负”这件事，在样本足够多时会被统计地纠正。</p>
<p>但现实中有三股力量让它变成真偏差：</p>
<ol>
<li><strong>方差大</strong>：<span class="hl"><strong>长篇回答各部分相对独立，互为噪声</strong></span>，单个方面的信用分配信噪比约 1/k，而一组只有 8–16 个样本。数学推理步步相依，没这么严重。</li>
<li><strong>补偿性聚合</strong>：<span class="hl"><strong>多满足几条可选标准就能抵消一条 Pitfall，目标的最优解本身就可能含错</strong></span>。这是目标设计问题，不是信用分配问题。</li>
<li><strong>伪相关</strong>（最隐蔽）：<span class="hl"><strong>梯度对某个特征的推动方向取决于它与奖励的协方差</strong></span>。幻觉常和“写得长”同时出现，而长回答满足更多条目，于是即使幻觉有害，协方差仍可能为正，它在期望意义上也被顺带强化。</li>
</ol>
<p>再叠加 GRPO 已有的长度偏差（<span class="hl"><strong>按 token 平均时长正样本贡献更多梯度；序列内除以长度时长负样本受罚更轻</strong></span>），rubric 对长度的偏好会被进一步放大。</p>
<p>下图对比了原版 GRPO 与加入局部负项后，advantage 如何落到回答的各个片段上：</p>
<figure><svg viewBox="0 0 760 290" role="img" aria-label="正的序列级 advantage 会连带强化回答中的错误片段" font-size="13"><text data-claude-text-id="title" x="24" y="34" font-size="15" font-weight="600" fill="var(--cds-text-primary)">正的序列级 advantage 会连带强化回答中的错误片段</text><text data-claude-text-id="subtitle" x="24" y="56" font-size="11.5" fill="var(--cds-text-secondary)">一个总分 0.8 的回答，切成 5 个片段</text><g data-claude-anchor="row-grpo"><text data-claude-text-id="row-a-label" x="24" y="108" font-weight="600" fill="var(--cds-text-primary)">原版 GRPO</text><rect x="120" y="78" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="240" y="78" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="360" y="78" width="104" height="52" rx="8" fill="var(--cds-chart-status-critical)" fill-opacity="0.12" stroke="var(--cds-chart-status-critical)" stroke-width="2"></rect><rect x="480" y="78" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="600" y="78" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="a-s1" x="172" y="100" text-anchor="middle" fill="var(--cds-text-primary)">结论</text><text data-claude-text-id="a-s1-v" x="172" y="119" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="a-s2" x="292" y="100" text-anchor="middle" fill="var(--cds-text-primary)">推理</text><text data-claude-text-id="a-s2-v" x="292" y="119" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="a-s3" x="412" y="100" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">事实错误</text><text data-claude-text-id="a-s3-v" x="412" y="119" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">+A，被强化</text><text data-claude-text-id="a-s4" x="532" y="100" text-anchor="middle" fill="var(--cds-text-primary)">补充例子</text><text data-claude-text-id="a-s4-v" x="532" y="119" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="a-s5" x="652" y="100" text-anchor="middle" fill="var(--cds-text-primary)">注意事项</text><text data-claude-text-id="a-s5-v" x="652" y="119" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text></g><g data-claude-anchor="row-local"><text data-claude-text-id="row-b-label" x="24" y="204" font-weight="600" fill="var(--cds-text-primary)">加局部负项</text><rect x="120" y="174" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="240" y="174" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="360" y="174" width="104" height="52" rx="8" fill="var(--cds-chart-status-good)" fill-opacity="0.12" stroke="var(--cds-chart-status-good)" stroke-width="2"></rect><rect x="480" y="174" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><rect x="600" y="174" width="104" height="52" rx="8" fill="none" stroke="var(--cds-chart-axis)" stroke-width="1.25"></rect><text data-claude-text-id="b-s1" x="172" y="196" text-anchor="middle" fill="var(--cds-text-primary)">结论</text><text data-claude-text-id="b-s1-v" x="172" y="215" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="b-s2" x="292" y="196" text-anchor="middle" fill="var(--cds-text-primary)">推理</text><text data-claude-text-id="b-s2-v" x="292" y="215" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="b-s3" x="412" y="196" text-anchor="middle" font-weight="600" fill="var(--cds-text-primary)">事实错误</text><text data-claude-text-id="b-s3-v" x="412" y="215" text-anchor="middle" font-size="11.5" fill="var(--cds-text-primary)">+A − A_local</text><text data-claude-text-id="b-s4" x="532" y="196" text-anchor="middle" fill="var(--cds-text-primary)">补充例子</text><text data-claude-text-id="b-s4-v" x="532" y="215" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text><text data-claude-text-id="b-s5" x="652" y="196" text-anchor="middle" fill="var(--cds-text-primary)">注意事项</text><text data-claude-text-id="b-s5-v" x="652" y="215" text-anchor="middle" font-size="11.5" fill="var(--cds-text-secondary)">+A</text></g><text data-claude-text-id="note" x="24" y="264" font-size="11.5" fill="var(--cds-text-secondary)">A_local 只作用在 judge 定位出的违规 token 上，其余片段照常更新</text></svg><figcaption>一个总分为正的回答被切成五个片段。原版 GRPO 把同一个正 advantage 分给所有片段，事实错误也被推高；<span class="hl"><strong>加入局部负项后</strong></span>，只有 judge 定位出的违规片段被单独压低，其余片段照常更新。</figcaption></figure>
<p>一个常见的说法是，结果奖励每个 episode 只给 advantage 提供 O(1) 比特的信息（<a href="https://thinkingmachines.ai/blog/lora/">Schulman et al., 2025</a>）。从这个角度看 rubric 有点讽刺：它本来能提供 k 个比特——哪一条过了、哪一条没过——却在聚合成标量时又被压回了 O(1)。</p>

<h2>怎么发现，怎么缓解</h2>
<p>如果让我自己从头训一个 rubric RL，我最先做的不会是调算法，而是把日志改掉。</p>
<p>大多数训练面板上只有一条平均奖励曲线，它在 rubric RL 里几乎总是漂亮地往上走：这恰恰是问题所在。<span class="hl"><strong>总分上升可能来自模型真的变好了，也可能来自它学会了多写几句、多提几个关键词</strong></span>。两种情况在总分曲线上长得一模一样，只有拆到条目层面才能分开。</p>
<p>所以我的建议是：<span class="hl"><strong>把每个条目的满足率当成一等公民来记录，就像监督学习里会分别看每个类别的准确率一样。</strong></span></p>
<p>有了逐条目的数据，再按有效权重算出梯度实际花在哪里，就可以对照下面的症状：</p>
<ul>
<li><strong>Essential 早早满分、Optional 满足率一路上升、回答变长</strong> → 在刷容易的条目</li>
<li><strong>大量题组内 std 很小，advantage 却和其他题一样大</strong> → 噪声被放大</li>
<li><strong>Pitfall 违反率不降反升</strong> → 补偿性聚合或伪相关在起作用</li>
<li><strong>训练用 judge 分数上升，独立评判者（更强、不同家族或不带 rubric 的整体评判）停滞或下降</strong> → 明确的 hacking</li>
</ul>
<p>发现问题之后，修起来反而有比较清楚的路径，因为每个症状都能对应回三次压缩中的某一次。</p>
<p>按层次来看：</p>
<ol>
<li><strong>题目层</strong>：给 std 设下限或改用 batch 级 std；过滤条件从“std=0”改为“组内极差小于阈值”；让 judge 重复判分估计噪声，按信噪比加权。</li>
<li><strong>条目层</strong>：退役已饱和条目，用成对比较补充有区分度的新条目（OnlineRubrics）；设计时去冗余、限制 Optional 总权重。</li>
<li><strong>聚合方式</strong>：用门控或乘性惩罚替代加权求和：Essential 未满足或触发 Pitfall 时总分直接压低，好处不再能抵消错误。</li>
<li><strong>Token 层</strong>：把丢掉的局部负信号找回来，见下。</li>
</ol>
<p>Token 层常见的三种做法：</p>
<ul>
<li><strong>局部负项</strong>：让 judge 定位违规片段，在序列级 advantage 之外给这些 token 加一个负项（即上文的图）；</li>
<li><strong>最小编辑 + DPO</strong>：<span class="hl"><strong>对错误回答做最小修正，在这对样本上做 DPO，相同 token 在概率比中抵消，梯度自动集中到错误处；</strong></span></li>
<li><strong>逐 token 蒸馏</strong>：用以 rubric 为条件的教师给出逐 token 信号，从根上绕开序列级广播。</li>
</ul>
<blockquote>
<p><strong>一点推测：为什么隐式聚合效果更好？</strong></p>
<p>RaR 发现让 judge 整体打分（隐式聚合），比逐条加权求和效果更好。我猜这可能是因为<span class="hl"><strong>整体评判天然是非补偿性的：人和模型都会觉得一个严重错误就毁了整个回答，而不是机械地加减</strong></span>。我没有验证过这一点，不想说得太满，但它和上面“门控优于加权求和”的思路是一致的。</p>
</blockquote>

<h2>Rubric RL 更像 RLHF，而不是 RLVR</h2>
<p>回头看，<span class="hl"><strong>rubric 给了我们比二元对错丰富得多的信号，但标准 GRPO 把这些结构一层层压扁了：先压成标量，再按组内方差重新分配题目权重，最后均匀地洒到每个 token</strong></span>。每一步单看都无害，叠在一起却恰好偏向奖励“写得多”而不是“写得对”。</p>
<p>这让我觉得，rubric RL 在性质上更接近 RLHF，而不是 RLVR。RLVR 的奖励偏差低，大家才敢去掉 KL 惩罚、放松信任域，用 GRPO 替代 PPO；RLHF 的奖励模型有偏，所以需要 KL 和保守的裁剪来防止过度优化错误的东西（这个对比受 <a href="https://nrehiew.github.io/blog/sft_rl_opd/">nrehiew 的博客</a> 启发）。Rubric 的 judge 同样是有偏的代理，却常常直接沿用 RLVR 那套激进的默认设置。</p>
<p>我觉得这个问题在 Agent 任务上会变得更尖锐。<span class="hl"><strong>Agent 的轨迹更长，可以“多做一点”的空间更大：多调用一次工具、多写一段说明、多做一步检查，都可能让某条 rubric 从不满足变成满足。</strong></span>同时，<span class="hl"><strong>轨迹里的单个危险操作更难被序列级奖励捕捉到</strong></span>。前面讲的三种失真，在 Agent 场景里都会被放大，而这恰恰是大家最希望用 rubric 来补足二元奖励的地方。</p>
<p>所以问题的形状比答案清楚：我们想要 rubric 的结构信息（多维、能定位到片段），RLVR 的低偏差，以及 GRPO 的简洁。三者兼得的算法具体长什么样，我还不知道。</p>

<h2>参考文献</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath（GRPO 提出）</a></li>
<li><a href="https://arxiv.org/abs/2503.20783">Understanding R1-Zero-Like Training（Dr. GRPO）</a></li>
<li><a href="https://arxiv.org/abs/2503.14476">DAPO</a></li>
<li><a href="https://arxiv.org/abs/2505.08775">HealthBench</a></li>
<li><a href="https://arxiv.org/abs/2507.17746">Rubrics as Rewards</a></li>
<li><a href="https://arxiv.org/abs/2507.18624">Checklists Are Better Than Reward Models（RLCF）</a></li>
<li><a href="https://arxiv.org/abs/2508.12790">Reinforcement Learning with Rubric Anchors（Rubicon）</a></li>
<li><a href="https://arxiv.org/abs/2510.07743">OpenRubrics</a></li>
<li><a href="https://arxiv.org/abs/2510.07284">Online Rubrics Elicitation from Pairwise Comparisons</a></li>
<li><a href="https://arxiv.org/abs/2507.20534">Kimi K2 Technical Report</a></li>
<li><a href="https://arxiv.org/abs/2605.12474">Reward Hacking in Rubric-Based Reinforcement Learning</a></li>
<li><a href="https://arxiv.org/abs/2306.01693">Fine-Grained Human Feedback</a></li>
<li><a href="https://arxiv.org/abs/2507.05386">Lai et al., 2025：RL 中的数据相关正则</a></li>
<li><a href="https://thinkingmachines.ai/blog/lora/">Schulman et al., 2025：LoRA Without Regret（结果奖励的信息量）</a></li>
<li><a href="https://nrehiew.github.io/blog/sft_rl_opd/">nrehiew：SFT, RL, and On-Policy Distillation Through a Distributional Lens</a></li>
</ul>
{% endraw %}
