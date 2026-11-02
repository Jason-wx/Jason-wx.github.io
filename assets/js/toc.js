// Build the sidebar table of contents from h2/h3/h4 in the post body.
(function () {
  var nav = document.getElementById('TableOfContents');
  var heads = document.querySelectorAll('#post-content > h2, #post-content > h3, #post-content > h4');
  if (!nav || !heads.length) return;
  var used = {};
  var root = document.createElement('ul'), stack = [{ level: 1, ul: root }];
  heads.forEach(function (h) {
    var level = +h.tagName[1];
    if (!h.id) {
      var id = h.textContent.toLowerCase().trim().replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-|-$/g, '') || 'section';
      while (used[id]) id += '-x';
      h.id = id;
    }
    used[h.id] = 1;
    while (stack.length > 1 && stack[stack.length - 1].level >= level) stack.pop();
    var parent = stack[stack.length - 1].ul;
    var li = document.createElement('li'), a = document.createElement('a');
    a.href = '#' + h.id; a.textContent = h.textContent;
    li.appendChild(a);
    var ul = document.createElement('ul');
    li.appendChild(ul);
    parent.appendChild(li);
    stack.push({ level: level, ul: ul });
  });
  root.querySelectorAll('ul:empty').forEach(function (u) { u.remove(); });
  nav.appendChild(root);
})();
