---
layout: default
title: 博客
permalink: /blog/
---
<h2 class="section-title" style="margin-top:0">博客</h2>
<ul class="post-list">
{% for post in site.posts %}{% if post.list != false %}
  <li><a href="{{ post.url | relative_url }}">
    <div class="pl-title">{{ post.title }}</div>
    <div class="pl-date">{{ post.date | date: "%Y-%m-%d" }}</div>
    {% if post.subtitle %}<div class="pl-sub">{{ post.subtitle }}</div>{% endif %}
  </a></li>
{% endif %}{% endfor %}
</ul>
