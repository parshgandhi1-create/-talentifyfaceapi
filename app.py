<?php
// ===================================
// LIST ALL IMAGES IN SCHOOL FOLDER
// ===================================

header("Content-Type: application/json");

$school_id = $_GET['school_id'] ?? '';
$folder = __DIR__ . "/../uploads/schools/" . $school_id;

if (!is_dir($folder)) {
    echo json_encode(["error" => "Folder not found"]);
    exit();
}

$files = array_values(array_filter(scandir($folder), function($file) use ($folder) {
    return !is_dir($folder . '/' . $file) && preg_match('/\.(jpg|jpeg|png)$/i', $file);
}));

echo json_encode($files);
?>
