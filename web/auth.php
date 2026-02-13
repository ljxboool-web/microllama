<?php
// auth.php - LOGIC CONTROLLER
require 'sys.php';

$act = $_GET['act'] ?? '';
$error = '';

// ----------------------------------------------------
// 登录处理
// ----------------------------------------------------
if ($act === 'login' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf_token($_POST['csrf_token'])) die("SECURITY ALERT: CSRF DETECTED");
    
    $username = clean_input($_POST['username']);
    $password = $_POST['password'];
    
    $stmt = $pdo->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->execute([$username]);
    $user = $stmt->fetch();
    
    if ($user && password_verify($password, $user['password'])) {
        // 更新登录时间
        $pdo->prepare("UPDATE users SET last_login = ? WHERE id = ?")->execute([time(), $user['id']]);
        
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['username'] = $user['username'];
        $_SESSION['role'] = $user['role'];
        $_SESSION['email'] = $user['email'];
        header("Location: index.php");
        exit;
    } else {
        header("Location: user.php?error=INVALID_CREDENTIALS");
        exit;
    }
}

// ----------------------------------------------------
// 注册处理
// ----------------------------------------------------
if ($act === 'register' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf_token($_POST['csrf_token'])) die("SECURITY ALERT: CSRF DETECTED");

    $username = clean_input($_POST['username']);
    $password = $_POST['password'];
    $email = clean_input($_POST['email']);
    
    // 简单验证
    if (strlen($username) < 3 || strlen($password) < 5) {
        header("Location: user.php?mode=reg&error=WEAK_PROTOCOL");
        exit;
    }
    
    try {
        $stmt = $pdo->prepare("INSERT INTO users (username, password, email, created_at, last_login) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute([
            $username, 
            password_hash($password, PASSWORD_DEFAULT),
            $email,
            time(),
            time()
        ]);
        
        // 自动登录
        $_SESSION['user_id'] = $pdo->lastInsertId();
        $_SESSION['username'] = $username;
        $_SESSION['role'] = 'user';
        header("Location: index.php");
        exit;
    } catch (PDOException $e) {
        header("Location: user.php?mode=reg&error=USER_EXISTS");
        exit;
    }
}

// ----------------------------------------------------
// 发帖处理
// ----------------------------------------------------
if ($act === 'post_thread' && isset($_SESSION['user_id'])) {
    if (!verify_csrf_token($_POST['csrf_token'])) die("SECURITY ALERT");
    $uid = (int)($_SESSION['user_id'] ?? 0);

    // 确认当前用户在用户表中存在，防止外键错误
    $check = $pdo->prepare("SELECT id FROM users WHERE id = ?");
    $check->execute([$uid]);
    if (!$check->fetchColumn()) {
        session_destroy();
        header("Location: user.php?error=AUTH_REQUIRED");
        exit;
    }

    $title = clean_input($_POST['title']);
    $content = trim($_POST['content']); // 允许部分 Markdown
    $uploaded_markdown = '';

    // 图片上传（可选，多张）
    if (!empty($_FILES['images']) && is_array($_FILES['images']['name'])) {
        $count = count($_FILES['images']['name']);
        for ($i = 0; $i < $count; $i++) {
            if (empty($_FILES['images']['name'][$i])) continue;
            $file = [
                'name' => $_FILES['images']['name'][$i],
                'type' => $_FILES['images']['type'][$i],
                'tmp_name' => $_FILES['images']['tmp_name'][$i],
                'error' => $_FILES['images']['error'][$i],
                'size' => $_FILES['images']['size'][$i],
            ];

            $saved = save_uploaded_image($file);
            if ($saved) {
                // 插入 Markdown 图片：![image](uploads/xxx.jpg)
                $uploaded_markdown .= "\n\n![" . clean_input(pathinfo($file['name'], PATHINFO_FILENAME)) . "](" . $saved . ")";
            }
        }
    }
    
    if ($title && $content) {
        $content .= $uploaded_markdown;
        $stmt = $pdo->prepare("INSERT INTO threads (user_id, title, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute([$uid, $title, $content, time(), time()]);
    }
    header("Location: index.php");
    exit;
}

// ----------------------------------------------------
// 回复处理
// ----------------------------------------------------
if ($act === 'reply' && isset($_SESSION['user_id'])) {
    if (!verify_csrf_token($_POST['csrf_token'])) die("SECURITY ALERT");
    
    $tid = (int)$_POST['thread_id'];
    $content = trim($_POST['content']);
    $uid = (int)($_SESSION['user_id'] ?? 0);

    if ($tid && $content) {
        // 确认帖子和用户都存在，防止外键错误
        $checkThread = $pdo->prepare("SELECT id FROM threads WHERE id = ?");
        $checkThread->execute([$tid]);
        $threadExists = (bool)$checkThread->fetchColumn();

        $checkUser = $pdo->prepare("SELECT id FROM users WHERE id = ?");
        $checkUser->execute([$uid]);
        $userExists = (bool)$checkUser->fetchColumn();

        if ($threadExists && $userExists) {
            // 插入评论
            $stmt = $pdo->prepare("INSERT INTO comments (thread_id, user_id, content, created_at) VALUES (?, ?, ?, ?)");
            $stmt->execute([$tid, $uid, $content, time()]);
            
            // 更新帖子计数和时间
            $pdo->prepare("UPDATE threads SET reply_count = reply_count + 1, updated_at = ? WHERE id = ?")
                ->execute([time(), $tid]);
        }
    }
    header("Location: view.php?id=$tid");
    exit;
}

// ----------------------------------------------------
// 退出处理
// ----------------------------------------------------
if ($act === 'logout') {
    session_destroy();
    header("Location: user.php");
    exit;
}

// ----------------------------------------------------
// 点赞：帖子
// ----------------------------------------------------
if ($act === 'toggle_like_thread' && isset($_SESSION['user_id']) && $_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf_token($_POST['csrf_token'])) {
        http_response_code(400);
        echo json_encode(['ok' => false, 'error' => 'CSRF_DETECTED']);
        exit;
    }
    $tid = (int)($_POST['thread_id'] ?? 0);
    if (!$tid) {
        http_response_code(400);
        echo json_encode(['ok' => false, 'error' => 'INVALID_INPUT']);
        exit;
    }

    header('Content-Type: application/json; charset=UTF-8');
    $uid = (int)$_SESSION['user_id'];

    try {
        $pdo->beginTransaction();
        $stmt = $pdo->prepare("SELECT 1 FROM thread_likes WHERE thread_id = ? AND user_id = ?");
        $stmt->execute([$tid, $uid]);
        $liked = (bool)$stmt->fetchColumn();

        if ($liked) {
            $pdo->prepare("DELETE FROM thread_likes WHERE thread_id = ? AND user_id = ?")->execute([$tid, $uid]);
            $pdo->prepare("UPDATE threads SET like_count = CASE WHEN like_count > 0 THEN like_count - 1 ELSE 0 END WHERE id = ?")->execute([$tid]);
            $liked = false;
        } else {
            $pdo->prepare("INSERT OR IGNORE INTO thread_likes (thread_id, user_id, created_at) VALUES (?, ?, ?)")->execute([$tid, $uid, time()]);
            $pdo->prepare("UPDATE threads SET like_count = like_count + 1 WHERE id = ?")->execute([$tid]);
            $liked = true;
        }

        $count = (int)$pdo->query("SELECT like_count FROM threads WHERE id = " . (int)$tid)->fetchColumn();
        $pdo->commit();
        echo json_encode(['ok' => true, 'liked' => $liked, 'count' => $count]);
        exit;
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) $pdo->rollBack();
        http_response_code(500);
        echo json_encode(['ok' => false, 'error' => 'SERVER_ERROR']);
        exit;
    }
}

// ----------------------------------------------------
// 点赞：评论
// ----------------------------------------------------
if ($act === 'toggle_like_comment' && isset($_SESSION['user_id']) && $_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf_token($_POST['csrf_token'])) {
        http_response_code(400);
        echo json_encode(['ok' => false, 'error' => 'CSRF_DETECTED']);
        exit;
    }
    $cid = (int)($_POST['comment_id'] ?? 0);
    if (!$cid) {
        http_response_code(400);
        echo json_encode(['ok' => false, 'error' => 'INVALID_INPUT']);
        exit;
    }

    header('Content-Type: application/json; charset=UTF-8');
    $uid = (int)$_SESSION['user_id'];

    try {
        $pdo->beginTransaction();
        $stmt = $pdo->prepare("SELECT 1 FROM comment_likes WHERE comment_id = ? AND user_id = ?");
        $stmt->execute([$cid, $uid]);
        $liked = (bool)$stmt->fetchColumn();

        if ($liked) {
            $pdo->prepare("DELETE FROM comment_likes WHERE comment_id = ? AND user_id = ?")->execute([$cid, $uid]);
            $pdo->prepare("UPDATE comments SET like_count = CASE WHEN like_count > 0 THEN like_count - 1 ELSE 0 END WHERE id = ?")->execute([$cid]);
            $liked = false;
        } else {
            $pdo->prepare("INSERT OR IGNORE INTO comment_likes (comment_id, user_id, created_at) VALUES (?, ?, ?)")->execute([$cid, $uid, time()]);
            $pdo->prepare("UPDATE comments SET like_count = like_count + 1 WHERE id = ?")->execute([$cid]);
            $liked = true;
        }

        $count = (int)$pdo->query("SELECT like_count FROM comments WHERE id = " . (int)$cid)->fetchColumn();
        $pdo->commit();
        echo json_encode(['ok' => true, 'liked' => $liked, 'count' => $count]);
        exit;
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) $pdo->rollBack();
        http_response_code(500);
        echo json_encode(['ok' => false, 'error' => 'SERVER_ERROR']);
        exit;
    }
}