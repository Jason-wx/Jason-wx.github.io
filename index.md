---
layout: default
---
<div class="intro">
<div class="avatar"><img src="{{ '/assets/images/life.jpg' | relative_url }}" alt="王潇"></div>
<div class="bio" markdown="1">

你好,我是王潇,东北大学人工智能专业硕士在读(2024.9 – 2027.6),我的课题组是 [NEUDM](https://neu-datamining.github.io/DM-Group-chinese.htm)。

我主要关注 **LLM Safety**、**Interpretability** 和 **Post-Training**:研究大模型为什么会在微调中丧失安全性,以及如何在数据和架构层面控制这种风险。

目前在上海人工智能实验室安全基模中心实习,做内生安全 MoE 架构和 MoE 遗忘学习。

我的 GitHub 是 [Jason-wx](https://github.com/Jason-wx),简历见[这里]({{ '/assets/resume.pdf' | relative_url }})。

</div>
</div>

<h2 class="section-title">Paper</h2>

<ul class="paper-list">
  <li>
    <a href="https://arxiv.org/abs/2605.04572">From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning</a>
    <div class="paper-venue">ICML 2026 · 第一作者</div>
    <div class="paper-desc">从参数动力学的角度分析大模型在无害数据微调中的安全退化,发现参数会沿危险方向持续漂移。基于这一机制,提出 SQSD 方法,在微调开始前为单个样本给出连续的风险分数,可用于微调数据的筛选。</div>
  </li>
  <li>
    <a href="https://arxiv.org/abs/2610.25791">Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into Removable Experts in Mixture-of-Experts Models</a>
    <div class="paper-venue">Tech Report · 上海人工智能实验室 · 核心贡献者</div>
    <div class="paper-desc">针对现有安全对齐与遗忘方法难以真正移除有害知识的问题,提出 REX:在 MoE 模型中新增专用于存放有害知识的专家,借助路由差异化与梯度掩码将有害知识迁移至这些专家,并在模型发布时将其整体移除。</div>
  </li>
</ul>

<h2 class="section-title">Blog</h2>

<p>记录自己的学习过程和思考。</p>

{% if site.posts.size == 0 %}
<p style="opacity:.6">还没有文章,敬请期待。</p>
{% else %}
<ul class="post-list">
{% for post in site.posts %}{% if post.list != false %}
  <li><a href="{{ post.url | relative_url }}">
    <div class="pl-title">{{ post.title }}</div>
    <div class="pl-date">{{ post.date | date: "%Y-%m-%d" }}</div>
    {% if post.subtitle %}<div class="pl-sub">{{ post.subtitle }}</div>{% endif %}
  </a></li>
{% endif %}{% endfor %}
</ul>
{% endif %}
