<?php
// index.php & publish.php combined logic
require 'sys.php';

// 路由判断：如果是 publish.php 或者 index.php?act=new
$is_publish = (strpos($_SERVER['SCRIPT_NAME'], 'publish.php') !== false);

render_head($is_publish ? t('new_transmission') : t('main_feed'));
render_nav();

if ($is_publish):
    // ---- 发布页面 ----
    if (!isset($_SESSION['user_id'])) {
        header("Location: user.php?error=AUTH_REQUIRED");
        exit;
    }
    ?>
    <div class="header-bar">
        <div class="brand">发布</div>
    </div>
    
    <form action="auth.php?act=post_thread" method="post" enctype="multipart/form-data">
        <input type="hidden" name="csrf_token" value="<?= generate_csrf_token() ?>">
        
        <div style="margin-bottom: 20px;">
            <label>标题</label>
            <input type="text" name="title" required placeholder="请输入标题" style="font-size: 1.1em;">
        </div>
        
        <div style="margin-bottom: 20px;">
            <label>内容</label>
            <textarea name="content" style="height: 260px;" required placeholder="请输入内容"></textarea>
        </div>

        <div style="margin-bottom: 20px;">
            <label>图片</label>
            <input type="file" name="images[]" accept=".jpg,.jpeg,.png,image/jpeg,image/png" multiple>
        </div>
        
        <div style="display:flex; gap: 10px; justify-content:flex-end;">
            <a href="index.php" class="action-btn" style="text-align:center;">取消</a>
            <button type="submit">发布</button>
        </div>
    </form>
    <?php

else:
    // ---- 首页列表 ----
    // 分页逻辑
    $page = isset($_GET['page']) ? (int)$_GET['page'] : 1;
    $per_page = 15;
    $offset = ($page - 1) * $per_page;

    $q = trim((string)($_GET['q'] ?? ''));
    $q_like = '%' . $q . '%';
    
    // 查询帖子
    if ($q !== '') {
        $sql = "SELECT t.*, u.username FROM threads t
                LEFT JOIN users u ON t.user_id = u.id
                WHERE t.title LIKE ? OR u.username LIKE ?
                ORDER BY t.is_sticky DESC, t.updated_at DESC
                LIMIT $per_page OFFSET $offset";
        $stmt = $pdo->prepare($sql);
        $stmt->execute([$q_like, $q_like]);
        $threads = $stmt->fetchAll();
    } else {
        $sql = "SELECT t.*, u.username FROM threads t 
                LEFT JOIN users u ON t.user_id = u.id 
                ORDER BY t.is_sticky DESC, t.updated_at DESC 
                LIMIT $per_page OFFSET $offset";
        $threads = $pdo->query($sql)->fetchAll();
    }
    ?>
    
    <div class="header-bar">
        <div class="brand">首页</div>
        <?php if(isset($_SESSION['user_id'])): ?>
            <a href="publish.php">发帖</a>
        <?php endif; ?>
    </div>

    <div class="card card--square card--tight" style="margin-bottom: 12px;">
        <form method="get" action="index.php" class="searchbar">
            <input type="text" name="q" value="<?= htmlspecialchars($q) ?>" placeholder="搜索 标题 或 作者">
            <button type="submit" style="width: 120px;">搜索</button>
        </form>
    </div>
    
    <div class="thread-list">
        <?php if(count($threads) == 0): ?>
            <div style="padding: 40px; text-align: center; color: var(--muted);">
                暂无内容
            </div>
        <?php endif; ?>

        <?php foreach($threads as $t): ?>
            <div class="card">
                <div class="card-head">
                    <img class="avatar" src="<?= get_avatar($t['username'] . '@local') ?>" alt="">
                    <div>
                        <div style="font-weight:700; color: var(--text);"><?= htmlspecialchars($t['username']) ?></div>
                        <div class="card-meta"><?= time_ago($t['updated_at']) ?></div>
                    </div>
                </div>
                <div class="card-title">
                    <a href="view.php?id=<?= $t['id'] ?>"><?= htmlspecialchars($t['title']) ?></a>
                </div>
                <div class="card-actions">
                    <a class="action-btn" href="view.php?id=<?= $t['id'] ?>" style="text-align:center;">赞 <?= (int)($t['like_count'] ?? 0) ?></a>
                    <a class="action-btn" href="view.php?id=<?= $t['id'] ?>" style="text-align:center;">评论 <?= (int)($t['reply_count'] ?? 0) ?></a>
                    <a class="action-btn" href="view.php?id=<?= $t['id'] ?>" style="text-align:center;">浏览 <?= (int)($t['views'] ?? 0) ?></a>
                </div>
            </div>
        <?php endforeach; ?>
    </div>
    
    <div class="pagination-bar" style="display: flex; justify-content: space-between; color: var(--muted);">
        <?php if($page > 1): ?>
            <a href="?page=<?= $page - 1 ?><?= $q !== '' ? '&q=' . urlencode($q) : '' ?>">上一页</a>
        <?php else: ?>
            <span>上一页</span>
        <?php endif; ?>
        
        <span>第 <?= (int)$page ?> 页</span>
        
        <?php if(count($threads) == $per_page): ?>
            <a href="?page=<?= $page + 1 ?><?= $q !== '' ? '&q=' . urlencode($q) : '' ?>">下一页</a>
        <?php else: ?>
            <span>下一页</span>
        <?php endif; ?>
    </div>
    <?php
endif;

render_footer();
?>