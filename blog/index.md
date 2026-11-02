---
layout: default
title: Blog
permalink: /blog/
---
<h2 class="section-title" style="margin-top:0">Writing</h2>
<ul class="post-list">
{% for post in site.posts %}
  <li><a href="{{ post.url | relative_url }}">
    <div class="pl-title">{{ post.title }}</div>
    <div class="pl-date">{{ post.date | date: "%B %-d, %Y" }}</div>
    {% if post.subtitle %}<div class="pl-sub">{{ post.subtitle }}</div>{% endif %}
  </a></li>
{% endfor %}
</ul>
