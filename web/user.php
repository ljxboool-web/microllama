<?php
require 'sys.php';
render_head(t('gateway'));
render_nav();

$mode = $_GET['mode'] ?? 'login';
$error = $_GET['error'] ?? '';
?>

<div class="header-bar">
    <div class="brand">登录注册</div>
</div>

<?php if($error): ?>
    <div style="border: 1px solid var(--term-alert); color: var(--term-alert); padding: 10px; margin-bottom: 20px;">
        <?= htmlspecialchars(error_message($error)) ?>
    </div>
<?php endif; ?>

<?php if(isset($_SESSION['user_id']) && ($_GET['act']??'') == 'profile'): 
    // ---- 个人资料页 ----
    $uid = $_SESSION['user_id'];
    $stmt = $pdo->prepare("SELECT COUNT(*) FROM threads WHERE user_id = ?");
    $stmt->execute([$uid]);
    $thread_count = $stmt->fetchColumn();
    
    $stmt = $pdo->prepare("SELECT COUNT(*) FROM comments WHERE user_id = ?");
    $stmt->execute([$uid]);
    $reply_count = $stmt->fetchColumn();
?>
    <div class="header-bar">
        <div class="brand">个人主页</div>
    </div>
    <div class="card card--square card--tight">
        <div style="display:flex; gap: 12px; align-items:center;">
            <img class="avatar avatar--square" src="<?= get_avatar($_SESSION['email']) ?>" alt="">
            <div style="min-width:0;">
                <div style="font-weight: 800; font-size: 16px; color: var(--text);"><?= htmlspecialchars($_SESSION['username']) ?></div>
                <div class="card-meta">编号 <?= (int)$uid ?></div>
            </div>
        </div>
        <div style="margin-top: 12px;" class="stat-grid">
            <div class="stat-box">
                <div class="stat-num"><?= (int)$thread_count ?></div>
                <div class="stat-label">发帖</div>
            </div>
            <div class="stat-box">
                <div class="stat-num"><?= (int)$reply_count ?></div>
                <div class="stat-label">评论</div>
            </div>
        </div>
    </div>

<?php else: ?>

    <div style="display:flex; justify-content:center; gap: 12px; margin-bottom: 16px;">
        <a href="?mode=login" class="<?= $mode=='login'?'':'dim' ?>">登录</a>
        <a href="?mode=reg" class="<?= $mode=='reg'?'':'dim' ?>">注册</a>
    </div>

    <div class="card" style="max-width: 420px; margin: 0 auto;">
        <form action="auth.php?act=<?= $mode=='login'?'login':'register' ?>" method="post">
            <input type="hidden" name="csrf_token" value="<?= generate_csrf_token() ?>">
            
            <label>用户名</label>
            <input type="text" name="username" required autocomplete="off">
            
            <label>密码</label>
            <input type="password" name="password" required>
            
            <?php if($mode == 'reg'): ?>
                <label>邮箱</label>
                <input type="email" name="email">
            <?php endif; ?>
            
            <button type="submit" style="width: 100%; margin-top: 10px;">
                <?= $mode=='login' ? '登录' : '注册' ?>
            </button>
        </form>
    </div>

<?php endif; ?>

<?php render_footer(); ?>