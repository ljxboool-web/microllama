<?php
require 'sys.php';

$tid = (int)($_GET['id'] ?? 0);
if (!$tid) header("Location: index.php");

// 获取帖子数据
$stmt = $pdo->prepare("SELECT t.*, u.username, u.email FROM threads t LEFT JOIN users u ON t.user_id = u.id WHERE t.id = ?");
$stmt->execute([$tid]);
$thread = $stmt->fetch();

if (!$thread) die("DATA_LOSS: Thread not found.");

// 增加浏览量
$pdo->prepare("UPDATE threads SET views = views + 1 WHERE id = ?")->execute([$tid]);

// 获取评论
$stmt = $pdo->prepare("SELECT c.*, u.username, u.email FROM comments c LEFT JOIN users u ON c.user_id = u.id WHERE c.thread_id = ? ORDER BY c.id ASC");
$stmt->execute([$tid]);
$comments = $stmt->fetchAll();

// 是否已点赞（帖子）
$thread_liked = false;
if (isset($_SESSION['user_id'])) {
    $stmt = $pdo->prepare("SELECT 1 FROM thread_likes WHERE thread_id = ? AND user_id = ?");
    $stmt->execute([$tid, (int)$_SESSION['user_id']]);
    $thread_liked = (bool)$stmt->fetchColumn();
}

// 评论已点赞集合
$liked_comments = [];
if (isset($_SESSION['user_id']) && count($comments) > 0) {
    $ids = array_map(fn($r) => (int)$r['id'], $comments);
    $placeholders = implode(',', array_fill(0, count($ids), '?'));
    $params = $ids;
    $params[] = (int)$_SESSION['user_id'];
    $stmt = $pdo->prepare("SELECT comment_id FROM comment_likes WHERE comment_id IN ($placeholders) AND user_id = ?");
    $stmt->execute($params);
    $liked_comments = array_flip(array_map('intval', $stmt->fetchAll(PDO::FETCH_COLUMN)));
}

render_head($thread['title']);
render_nav();
?>

<div style="margin-bottom: 10px;">
    <a href="index.php">返回首页</a>
</div>

<div class="card">
    <div class="card-head">
        <img class="avatar" src="<?= get_avatar($thread['email']) ?>" alt="">
        <div>
            <div style="font-weight:800; color: var(--text);"><?= htmlspecialchars($thread['username']) ?></div>
            <div class="card-meta"><?= time_ago($thread['created_at']) ?></div>
        </div>
    </div>
    <div class="card-title"><?= htmlspecialchars($thread['title']) ?></div>
    <div style="color: var(--text); line-height: 1.7;">
        <?= simple_markdown($thread['content']) ?>
    </div>
    <div class="card-actions">
        <button type="button" class="action-btn" onclick="toggleLikeThread(<?= (int)$tid ?>, this)">
            赞 <span class="like-count"><?= (int)($thread['like_count'] ?? 0) ?></span>
        </button>
        <a class="action-btn" href="#commentBox" style="text-align:center; display:block;">评论 <?= (int)count($comments) ?></a>
        <div class="action-btn" style="text-align:center;"><?= (int)$thread['views'] ?> 浏览</div>
    </div>
</div>

<div class="header-bar" style="margin-top: 18px;">
    <div class="brand">评论</div>
</div>

<?php foreach ($comments as $index => $c): ?>
    <div class="card">
        <div class="card-head">
            <img class="avatar" src="<?= get_avatar($c['email']) ?>" alt="">
            <div style="flex:1;">
                <div style="display:flex; justify-content: space-between; gap: 10px; align-items:center;">
                    <div>
                        <div style="font-weight:800; color: var(--text);"><?= htmlspecialchars($c['username']) ?></div>
                        <div class="card-meta"><?= time_ago($c['created_at']) ?></div>
                    </div>
                    <button type="button" class="action-btn" style="max-width: 140px;" onclick="toggleLikeComment(<?= (int)$c['id'] ?>, this)">
                        赞 <span class="like-count"><?= (int)($c['like_count'] ?? 0) ?></span>
                    </button>
                </div>
                <div style="margin-top: 10px; color: var(--text); line-height: 1.7;"><?= nl2br(htmlspecialchars($c['content'])) ?></div>
            </div>
        </div>
    </div>
<?php endforeach; ?>

<?php if(isset($_SESSION['user_id'])): ?>
    <div id="commentBox" style="margin-top: 18px;">
        <form action="auth.php?act=reply" method="post">
            <input type="hidden" name="csrf_token" value="<?= generate_csrf_token() ?>">
            <input type="hidden" name="thread_id" value="<?= $tid ?>">
            <label>发表评论</label>
            <textarea name="content" style="height: 110px;" required placeholder="请输入评论"></textarea>
            <button type="submit">发布</button>
        </form>
    </div>
<?php else: ?>
    <div style="margin-top: 18px; padding: 14px; text-align: center; color: var(--term-alert);">
        请先登录
    </div>
<?php endif; ?>

<?php render_footer(); ?>

<script>
async function postJson(url, data) {
  const form = new FormData();
  for (const k in data) form.append(k, data[k]);
  const res = await fetch(url, { method: 'POST', body: form, credentials: 'same-origin' });
  return await res.json();
}

async function toggleLikeThread(threadId, btn) {
  const csrf = '<?= generate_csrf_token() ?>';
  const r = await postJson('auth.php?act=toggle_like_thread', { csrf_token: csrf, thread_id: threadId });
  if (!r || !r.ok) return;
  btn.querySelector('.like-count').textContent = String(r.count);
}

async function toggleLikeComment(commentId, btn) {
  const csrf = '<?= generate_csrf_token() ?>';
  const r = await postJson('auth.php?act=toggle_like_comment', { csrf_token: csrf, comment_id: commentId });
  if (!r || !r.ok) return;
  btn.querySelector('.like-count').textContent = String(r.count);
}
</script>