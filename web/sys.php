<?php
// sys.php - THE KERNEL
// ==========================================
// 核心系统定义文件，包含数据库层与渲染层
// ==========================================

// 更安全的 Session Cookie 参数（尽量在 session_start() 之前）
$is_https = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
    || (!empty($_SERVER['SERVER_PORT']) && (int)$_SERVER['SERVER_PORT'] === 443);
session_set_cookie_params([
    'lifetime' => 0,
    'path' => '/',
    'secure' => $is_https,
    'httponly' => true,
    'samesite' => 'Lax',
]);

session_start();
ob_start();
date_default_timezone_set('Asia/Shanghai');

// 基础安全响应头
header('X-Frame-Options: DENY');
header('X-Content-Type-Options: nosniff');
header('Referrer-Policy: same-origin');

// 1. 系统配置常量
define('SYS_NAME', '微博社区');
define('SYS_VER', '2.5.0-Release');
define('DB_FILE', __DIR__ . '/core_data.db');
define('SALT', 'X_9982_GEO_KEY_#@!'); // 盐值

// 多语言配置：支持 en / zh，可按需扩展
$SUPPORTED_LANGS = ['en', 'zh'];

// 文案字典
$LANG_STRINGS = [
    'en' => [
        'main_feed' => 'Main Feed',
        'new_transmission' => 'New Transmission',
        'my_node' => 'My Node',
        'terminate_session' => 'Terminate Session',
        'authenticate' => 'Authenticate',
        'navigation' => 'Navigation',
        'sys_info' => 'System Info',
        'user' => 'User',
        'role' => 'Role',
        'time' => 'Time',
        'status_online' => 'STATUS: ONLINE',
        'gateway' => 'Gateway Interface',
        'error_log' => 'ERROR LOG',
        'auth_login' => 'Auth Login',
        'new_identity' => 'New Identity',
        'username' => 'Username',
        'password' => 'Password',
        'email' => 'Email (For Avatar)',
        'execute_login' => 'Execute Login',
        'execute_registration' => 'Execute Registration',
        'return_root' => 'Return to Root',
        'data_stream' => 'Data Stream',
        'append_packet' => 'Append Data Packet',
        'transmit' => 'Transmit',
        'access_denied' => 'Access denied: authentication required to reply',
        'public_feed' => 'Public Feed',
        'no_packets' => 'No data packets found',
        'prev_page' => 'Prev Page',
        'next_page' => 'Next Page',
        'page' => 'Page',
        'author' => 'Author',
        'updated' => 'Updated',
        'replies' => 'Replies',
        'views' => 'Views',
        'op' => 'OP',
        'created_time' => 'Time',
        'packet_id' => 'Packet ID',
        'just_now' => 'Just now',
        'mins_ago' => '%s mins ago',
        'hrs_ago' => '%s hrs ago',
        'weak_protocol' => 'Username or password is too short.',
        'invalid_credentials' => 'Invalid username or password.',
        'user_exists' => 'Username already exists.',
        'auth_required' => 'Please login first.',
        'csrf_detected' => 'Security alert: invalid request.',
        'invalid_input' => 'Invalid input.',
        'subject_header' => 'Subject Header',
        'payload_content' => 'Payload Content (Markdown Supported)',
        'cancel' => 'Cancel',
        'upload' => 'Upload to Network',
        'lang_en' => 'English',
        'lang_zh' => '中文',
        'theme_light' => 'Light',
        'theme_dark' => 'Dark',
        'lang_label' => 'Language',
        'theme_label' => 'Theme',
    ],
    'zh' => [
        'main_feed' => '主时间线',
        'new_transmission' => '发布新帖',
        'my_node' => '我的节点',
        'terminate_session' => '退出登录',
        'authenticate' => '登录 / 注册',
        'navigation' => '导航',
        'sys_info' => '系统信息',
        'user' => '用户',
        'role' => '角色',
        'time' => '时间',
        'status_online' => '状态：在线',
        'gateway' => '访问网关',
        'error_log' => '错误日志',
        'auth_login' => '登录',
        'new_identity' => '注册新身份',
        'username' => '用户名',
        'password' => '密码',
        'email' => '邮箱（用于头像）',
        'execute_login' => '执行登录',
        'execute_registration' => '执行注册',
        'return_root' => '返回主页',
        'data_stream' => '数据流',
        'append_packet' => '追加数据包',
        'transmit' => '发送',
        'access_denied' => '未登录，无法回复',
        'public_feed' => '公共信息流',
        'no_packets' => '暂无数据包',
        'prev_page' => '上一页',
        'next_page' => '下一页',
        'page' => '第 %s 页',
        'author' => '作者',
        'updated' => '更新于',
        'replies' => '回复',
        'views' => '浏览',
        'op' => '楼主',
        'created_time' => '时间',
        'packet_id' => '数据包ID',
        'just_now' => '刚刚',
        'mins_ago' => '%s 分钟前',
        'hrs_ago' => '%s 小时前',
        'weak_protocol' => '用户名或密码过短。',
        'invalid_credentials' => '用户名或密码错误。',
        'user_exists' => '用户名已存在。',
        'auth_required' => '请先登录。',
        'csrf_detected' => '安全警报：请求无效。',
        'invalid_input' => '输入不合法。',
        'subject_header' => '主题标题',
        'payload_content' => '内容正文（支持 Markdown）',
        'cancel' => '取消',
        'upload' => '上传到网络',
        'lang_en' => 'English',
        'lang_zh' => '中文',
        'theme_light' => '日间',
        'theme_dark' => '夜间',
        'lang_label' => '语言',
        'theme_label' => '主题',
    ],
];

function get_lang() {
    // 强制中文
    return 'zh';
}

function t($key) {
    global $LANG_STRINGS;
    $lang = get_lang();
    return $LANG_STRINGS[$lang][$key] ?? ($LANG_STRINGS['en'][$key] ?? $key);
}

function error_message($code) {
    // 将后端 error code 映射为可翻译文本
    $map = [
        'WEAK_PROTOCOL' => 'weak_protocol',
        'INVALID_CREDENTIALS' => 'invalid_credentials',
        'USER_EXISTS' => 'user_exists',
        'AUTH_REQUIRED' => 'auth_required',
        'CSRF_DETECTED' => 'csrf_detected',
        'INVALID_INPUT' => 'invalid_input',
    ];
    $key = $map[$code] ?? null;
    return $key ? t($key) : $code;
}

// 2. 数据库连接与自动初始化
try {
    $db_exists = file_exists(DB_FILE);
    $pdo = new PDO("sqlite:" . DB_FILE);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);

    // 启用外键约束
    $pdo->exec("PRAGMA foreign_keys = ON");

    if (!$db_exists) {
        // 建表：用户
        $pdo->exec("CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            avatar TEXT,
            role TEXT DEFAULT 'user',
            created_at INTEGER,
            last_login INTEGER
        )");

        // 建表：帖子
        $pdo->exec("CREATE TABLE threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            like_count INTEGER DEFAULT 0,
            is_sticky INTEGER DEFAULT 0,
            created_at INTEGER,
            updated_at INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )");

        // 建表：评论
        $pdo->exec("CREATE TABLE comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            user_id INTEGER,
            content TEXT NOT NULL,
            created_at INTEGER,
            like_count INTEGER DEFAULT 0,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )");

        // 点赞表：帖子点赞
        $pdo->exec("CREATE TABLE thread_likes (
            thread_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at INTEGER,
            PRIMARY KEY(thread_id, user_id),
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )");

        // 点赞表：评论点赞
        $pdo->exec("CREATE TABLE comment_likes (
            comment_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at INTEGER,
            PRIMARY KEY(comment_id, user_id),
            FOREIGN KEY(comment_id) REFERENCES comments(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )");
        
        // 插入管理员账号 (admin / admin888)
        $admin_pass = password_hash('admin888', PASSWORD_DEFAULT);
        $ts = time();
        $pdo->exec("INSERT INTO users (username, password, role, created_at) 
                    VALUES ('admin', '$admin_pass', 'admin', $ts)");
    }
} catch (PDOException $e) {
    die("KERNEL PANIC: Database initialization failed. " . $e->getMessage());
}

// 兼容已存在数据库：自动补齐点赞字段/表
function ensure_schema($pdo) {
    $exists = function($name) use ($pdo) {
        $stmt = $pdo->prepare("SELECT name FROM sqlite_master WHERE type IN ('table','index') AND name = ?");
        $stmt->execute([$name]);
        return (bool)$stmt->fetchColumn();
    };
    $colExists = function($table, $col) use ($pdo) {
        $rows = $pdo->query("PRAGMA table_info($table)")->fetchAll();
        foreach ($rows as $r) {
            if (($r['name'] ?? '') === $col) return true;
        }
        return false;
    };

    if ($exists('threads') && !$colExists('threads', 'like_count')) {
        $pdo->exec("ALTER TABLE threads ADD COLUMN like_count INTEGER DEFAULT 0");
    }
    if ($exists('comments') && !$colExists('comments', 'like_count')) {
        $pdo->exec("ALTER TABLE comments ADD COLUMN like_count INTEGER DEFAULT 0");
    }
    if (!$exists('thread_likes')) {
        $pdo->exec("CREATE TABLE thread_likes (
            thread_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at INTEGER,
            PRIMARY KEY(thread_id, user_id),
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )");
    }
    if (!$exists('comment_likes')) {
        $pdo->exec("CREATE TABLE comment_likes (
            comment_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at INTEGER,
            PRIMARY KEY(comment_id, user_id),
            FOREIGN KEY(comment_id) REFERENCES comments(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )");
    }
}

ensure_schema($pdo);

// 3. 安全与工具函数库

function clean_input($data) {
    return htmlspecialchars(trim($data), ENT_QUOTES, 'UTF-8');
}

function save_uploaded_image($file) {
    // 返回可访问的相对路径，如 uploads/202601/xxxx.jpg；失败返回 null
    if (empty($file) || !isset($file['error'])) return null;
    if ($file['error'] !== UPLOAD_ERR_OK) return null;
    if (empty($file['tmp_name']) || !is_uploaded_file($file['tmp_name'])) return null;

    // 限制大小：5MB
    $max_bytes = 5 * 1024 * 1024;
    if (!empty($file['size']) && (int)$file['size'] > $max_bytes) return null;

    // 判断图片类型 仅允许 jpg png
    $allowed = [
        'image/jpeg' => 'jpg',
        'image/png' => 'png',
    ];

    $mime = '';
    if (class_exists('finfo')) {
        try {
            $finfo = new finfo(FILEINFO_MIME_TYPE);
            $mime = $finfo->file($file['tmp_name']) ?: '';
        } catch (Throwable $e) {
            $mime = '';
        }
    }

    // 服务器没启用 finfo 时，使用 exif_imagetype 兜底
    if (!$mime && function_exists('exif_imagetype')) {
        $t = @exif_imagetype($file['tmp_name']);
        if ($t === IMAGETYPE_JPEG) $mime = 'image/jpeg';
        if ($t === IMAGETYPE_PNG) $mime = 'image/png';
    }

    if (!isset($allowed[$mime])) return null;

    $subdir = date('Ym');
    $base_dir = __DIR__ . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $subdir;
    if (!is_dir($base_dir)) {
        if (!mkdir($base_dir, 0755, true) && !is_dir($base_dir)) return null;
    }

    $filename = bin2hex(random_bytes(16)) . '.' . $allowed[$mime];
    $dest_abs = $base_dir . DIRECTORY_SEPARATOR . $filename;
    // 确保可写，避免线上直接崩溃
    if (!is_writable($base_dir)) return null;
    if (!move_uploaded_file($file['tmp_name'], $dest_abs)) return null;

    return 'uploads/' . $subdir . '/' . $filename;
}

function sanitize_img_src($src) {
    $src = trim((string)$src);
    if ($src === '') return '';
    // 允许本站 uploads 路径，或 http(s) 外链
    if (preg_match('#^uploads/[a-zA-Z0-9/_\\-\\.]+$#', $src)) return $src;
    if (preg_match('#^https?://#i', $src)) return $src;
    return '';
}

function generate_csrf_token() {
    if (empty($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

function verify_csrf_token($token) {
    return isset($_SESSION['csrf_token']) && hash_equals($_SESSION['csrf_token'], $token);
}

function time_ago($timestamp) {
    $diff = time() - $timestamp;
    if ($diff < 60) return t('just_now');
    if ($diff < 3600) return sprintf(t('mins_ago'), floor($diff / 60));
    if ($diff < 86400) return sprintf(t('hrs_ago'), floor($diff / 3600));
    return date('Y-m-d H:i', $timestamp);
}

function simple_markdown($text) {
    // 简易 Markdown 解析器
    $text = htmlspecialchars($text); // 先转义
    // Images: ![alt](url)
    $text = preg_replace_callback('/!\[(.*?)\]\((.*?)\)/', function ($m) {
        $alt = htmlspecialchars($m[1], ENT_QUOTES, 'UTF-8');
        $src = sanitize_img_src(htmlspecialchars_decode($m[2], ENT_QUOTES));
        if (!$src) return $m[0];
        return '<img src="' . htmlspecialchars($src, ENT_QUOTES, 'UTF-8') . '" alt="' . $alt . '" class="post-image">';
    }, $text);
    $text = preg_replace('/\*\*(.*?)\*\*/', '<strong>$1</strong>', $text); // Bold
    $text = preg_replace('/`(.*?)`/', '<code class="inline-code">$1</code>', $text); // Inline Code
    $text = preg_replace('/> (.*?)\n/', '<blockquote>$1</blockquote>', $text); // Quote
    $text = nl2br($text);
    return $text;
}

function get_avatar($email) {
    $hash = md5(strtolower(trim($email)));
    return "https://www.gravatar.com/avatar/$hash?d=retro&s=64";
}

// 4. UI 渲染引擎

function render_head($title = "UPLINK") {
    $token = generate_csrf_token();
    $current_lang = get_lang();
    $current_theme = $_COOKIE['theme'] ?? 'dark';
    ?>
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title><?= htmlspecialchars($title) ?> <?= SYS_NAME ?></title>
        <style>
            :root {
                --font-stack: 'Courier New', Courier, monospace;
                --radius-lg: 18px;
                --radius-md: 12px;
                --shadow-lg: 0 20px 60px rgba(0,0,0,.25);
                --shadow-md: 0 14px 40px rgba(0,0,0,.18);
            }

            :root[data-theme="dark"] {
                --bg-color: #0b0c10;
                --text: rgba(255,255,255,.92);
                --muted: rgba(235,235,245,.60);
                --panel-bg: #14161a;
                --border-color: rgba(255,255,255,.10);
                --term-green: #ff8200;
                --term-dim: rgba(235,235,245,.55);
                --term-alert: #ff3b30;
                --term-blue: #ff8200;
            }

            :root[data-theme="light"] {
                --bg-color: #f5f5f7;
                --text: rgba(28,28,30,.94);
                --muted: rgba(60,60,67,.60);
                --panel-bg: #ffffff;
                --border-color: rgba(0,0,0,.08);
                --term-green: #ff8200;
                --term-dim: rgba(60,60,67,.55);
                --term-alert: #ff3b30;
                --term-blue: #ff8200;
            }
            * { box-sizing: border-box; }
            body { 
                background: var(--bg-color);
                color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", system-ui, Segoe UI, Roboto, Helvetica, Arial;
                margin: 0; padding: 0;
                line-height: 1.5;
                font-size: 16px;
            }
            a { color: var(--term-blue); text-decoration: none; transition: 0.15s ease; }
            a:hover { color: var(--term-blue); text-decoration: underline; }
            
            /* 微博风布局 */
            .topbar {
                position: sticky;
                top: 0;
                z-index: 50;
                background: rgba(255,255,255,.92);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(0,0,0,.06);
            }
            :root[data-theme="dark"] .topbar {
                background: rgba(20,22,26,.92);
                border-bottom: 1px solid rgba(255,255,255,.08);
            }
            .topbar-inner {
                max-width: 1040px;
                margin: 0 auto;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
            }
            .logo {
                font-weight: 800;
                letter-spacing: .2px;
                color: var(--text);
            }
            .layout {
                max-width: 1180px;
                margin: 0 auto;
                padding: 24px 16px 40px;
                display: grid;
                grid-template-columns: 280px 1fr;
                gap: 20px;
            }
            @media(max-width: 900px) { .layout { grid-template-columns: 1fr; } }

            .pagination-bar {
                position: sticky;
                bottom: 0;
                margin-top: 16px;
                padding-top: 12px;
                padding-bottom: 12px;
                background: linear-gradient(to top, var(--panel-bg) 70%, rgba(255,255,255,0));
            }
            :root[data-theme="dark"] .pagination-bar {
                background: linear-gradient(to top, var(--panel-bg) 70%, rgba(20,22,26,0));
            }
            
            /* 组件样式 */
            .sidebar { border: 1px solid var(--border-color); padding: 14px; background: var(--panel-bg); height: fit-content; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
            .main-panel { border: 1px solid var(--border-color); padding: 16px; min-height: 80vh; background: var(--panel-bg); border-radius: 12px; box-shadow: 0 18px 40px rgba(0,0,0,.12); }
            
            /* 标题栏 */
            .header-bar { border-bottom: 1px solid rgba(0,0,0,.06); margin-bottom: 16px; padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
            :root[data-theme="dark"] .header-bar { border-bottom: 1px solid rgba(255,255,255,.10); }
            .brand { font-size: 1.1em; font-weight: 700; letter-spacing: 0.2px; color: var(--text); }
            
            /* 列表与卡片 */
            .thread-item { border-bottom: 1px solid rgba(0,0,0,.04); padding: 12px 0; display: flex; justify-content: space-between; align-items: center; }
            .thread-item:hover { background: rgba(0,0,0,.02); border-radius: 12px; }
            .thread-stat { font-size: 0.8em; color: var(--term-dim); text-align: right; min-width: 100px; }
            .badge { background-color: rgba(0,0,0,.06); color: var(--muted); padding: 3px 8px; font-size: 0.7em; margin-right: 6px; border-radius: 999px; font-weight: 600; }
            .sticky { color: var(--term-alert); border-left: 3px solid var(--term-alert); padding-left: 10px; }
            .card { border: 1px solid var(--border-color); background: var(--panel-bg); border-radius: 16px; padding: 14px; }
            .card + .card { margin-top: 12px; }
            .card-head { display: flex; gap: 10px; align-items: center; }
            .avatar { width: 40px; height: 40px; border-radius: 999px; border: 1px solid var(--border-color); }
            .card-meta { color: var(--muted); font-size: 13px; }
            .card-title { font-weight: 700; margin-top: 8px; margin-bottom: 8px; color: var(--text); }
            .card-actions { display: flex; gap: 10px; margin-top: 10px; }
            .action-btn { flex: 1; background: rgba(118,118,128,.12); color: var(--text); border: 0; border-radius: 12px; padding: 10px 12px; font-weight: 600; box-shadow: none; }
            .action-btn:hover { background: rgba(118,118,128,.18); }

            /* 更方正更简洁 */
            .card--square { border-radius: 10px; padding: 12px; }
            .card--tight { padding: 10px 12px; }
            .avatar--square { border-radius: 10px; }
            .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .stat-box { border: 1px solid var(--border-color); background: rgba(118,118,128,.10); border-radius: 10px; padding: 10px; text-align: center; }
            .stat-num { font-size: 18px; font-weight: 800; color: var(--text); }
            .stat-label { color: var(--muted); font-size: 13px; }

            .searchbar { display:flex; gap: 10px; align-items:center; }
            .searchbar input { margin-bottom: 0; }
            .searchbar button { box-shadow: none; }
            
            /* 表单 */
            input, textarea, select {
                background: rgba(118,118,128,.12);
                border: 1px solid var(--border-color);
                color: var(--text);
                width: 100%;
                padding: 10px 12px;
                font-family: inherit;
                margin-bottom: 10px;
                border-radius: 12px;
            }
            input::placeholder, textarea::placeholder { color: var(--muted); }
            input:focus, textarea:focus, select:focus { outline: 2px solid rgba(10, 132, 255, .4); outline-offset: 1px; border-color: rgba(10,132,255,.8); }
            button {
                background-color: var(--term-blue);
                color: #ffffff;
                border: 0;
                padding: 10px 16px;
                cursor: pointer;
                font-weight: 600;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,.20);
                text-transform: none;
            }
            button:hover { filter: brightness(1.03); }
            .like-btn {
                border: 0;
                background: rgba(120,120,128,.12);
                color: var(--text);
                padding: 6px 10px;
                border-radius: 999px;
                text-transform: none;
                box-shadow: none;
            }
            .like-btn:hover { background: rgba(120,120,128,.18); }
            
            /* 引用与代码 */
            blockquote { border-left: 4px solid rgba(0,0,0,.10); margin: 10px 0; padding-left: 15px; color: var(--muted); }
            .inline-code { background: rgba(118,118,128,.18); padding: 2px 6px; color: var(--text); border-radius: 6px; }
            .post-image { max-width: 100%; height: auto; display: block; margin: 10px 0; border: 1px solid var(--border-color); }
            
            /* 闪烁光标 */
            .cursor::after { content: ''; }
            @keyframes blink { 50% { opacity: 0; } }

            .toolbar {
                display: flex;
                gap: 8px;
                font-size: 0.75em;
                align-items: center;
            }
            .toolbar select {
                background: rgba(255,255,255,.10);
                border: 1px solid var(--border-color);
                color: var(--text);
                font-family: inherit;
                padding: 6px 8px;
                border-radius: 12px;
            }

            /* 移动端增强 */
            @media(max-width: 768px) {
                .layout { padding: 12px; gap: 12px; }
                .main-panel { padding: 14px; }
                .sidebar { padding: 12px; }
                .header-bar { flex-wrap: wrap; gap: 10px; align-items: center; }
                .brand { font-size: 1.2em; letter-spacing: 1px; }
                button { width: 100%; padding: 10px 14px; }
                input, textarea { font-size: 16px; } /* 避免 iOS 自动缩放 */
                .thread-item { flex-direction: column; align-items: flex-start; gap: 10px; }
                .thread-stat { text-align: left; min-width: 0; }
                .toolbar { flex-wrap: wrap; }
                .toolbar select { width: 100%; padding: 6px 8px; }
            }
        </style>
        <script>
            (function() {
                var theme = '<?= htmlspecialchars($current_theme) ?>';
                if (!theme) theme = 'dark';
                document.documentElement.setAttribute('data-theme', theme);
            })();

            function setTheme(theme) {
                document.documentElement.setAttribute('data-theme', theme);
                document.cookie = 'theme=' + theme + ';path=/;max-age=' + (60*60*24*365);
            }
        </script>
    </head>
    <body>
    <?php
}

function render_nav() {
    $u = $_SESSION['username'] ?? '游客';
    $role = $_SESSION['role'] ?? '游客';
    $lang = get_lang();
    $theme = $_COOKIE['theme'] ?? 'dark';
    ?>
    <div class="topbar">
        <div class="topbar-inner">
            <div class="logo">微博社区</div>
            <div style="display:flex; gap: 14px; align-items:center;">
                <a href="index.php">首页</a>
                <a href="publish.php">发布</a>
                <a href="user.php?act=profile">个人主页</a>
            </div>
            <div style="display:flex; gap: 10px; align-items:center;">
                <select onchange="setTheme(this.value);" style="width: 120px;">
                    <option value="light" <?= $theme === 'light' ? 'selected' : '' ?>>日间</option>
                    <option value="dark" <?= $theme === 'dark' ? 'selected' : '' ?>>夜间</option>
                </select>
                <?php if(isset($_SESSION['user_id'])): ?>
                    <a href="auth.php?act=logout" style="color: var(--term-alert);">退出</a>
                <?php else: ?>
                    <a href="user.php">登录</a>
                <?php endif; ?>
            </div>
        </div>
    </div>

    <div class="layout">
        <aside class="sidebar">
            <div style="display:flex; align-items:center; gap: 10px; margin-bottom: 14px;">
                <div style="font-weight: 700; color: var(--text);"><?= htmlspecialchars($u) ?></div>
                <div style="color: var(--muted); font-size: 13px;"><?= htmlspecialchars($role) ?></div>
            </div>
            <div style="color: var(--muted); font-size: 13px;">
                欢迎使用微博社区
            </div>
        </aside>
        <main class="main-panel">
    <?php
}

function render_footer() {
    ?>
        </main>
    </div> </body>
    </html>
    <?php
    ob_end_flush();
}
?>