$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$entries = New-Object System.Collections.Generic.List[object]
$successful = New-Object System.Collections.Generic.List[string]
$errors = @{}
$shell = New-Object -ComObject WScript.Shell
$namespaceShell = New-Object -ComObject Shell.Application

foreach ($source in @('start_menu_user', 'start_menu_common')) {
    try {
        $folderKind = if ($source -eq 'start_menu_user') { 'Programs' } else { 'CommonPrograms' }
        $root = [Environment]::GetFolderPath($folderKind)
        if ($root -and (Test-Path -LiteralPath $root)) {
            foreach ($file in Get-ChildItem -LiteralPath $root -Filter '*.lnk' -Recurse -File) {
                try {
                    $shortcut = $shell.CreateShortcut($file.FullName)
                    $appId = ''
                    try {
                        $directory = $namespaceShell.Namespace($file.DirectoryName)
                        $shellItem = $directory.ParseName($file.Name)
                        $appId = $shellItem.ExtendedProperty('System.AppUserModel.ID')
                    } catch { }
                    $entries.Add([ordered]@{
                        name = $file.BaseName
                        source = $source
                        source_key = $source + ':' + $file.FullName.ToLowerInvariant()
                        launch_type = 'shortcut'
                        launch_target = $file.FullName
                        resolved_target = $shortcut.TargetPath
                        raw_arguments = $shortcut.Arguments
                        working_directory = $shortcut.WorkingDirectory
                        icon_target = $shortcut.TargetPath
                        description = $shortcut.Description
                        app_id = $appId
                    })
                    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shortcut)
                } catch {
                    $errors[$source + ':' + $file.BaseName] = 'Не удалось прочитать ярлык'
                }
            }
        }
        $successful.Add($source)
    } catch {
        $errors[$source] = 'Не удалось прочитать меню Пуск'
    }
}

try {
    foreach ($app in Get-StartApps) {
        $entries.Add([ordered]@{
            name = $app.Name
            source = 'start_apps'
            source_key = 'start_apps:' + $app.AppID
            launch_type = 'app_id'
            launch_target = $app.AppID
        })
    }
    $successful.Add('start_apps')
} catch {
    $errors['start_apps'] = 'Не удалось получить список приложений Windows'
}

[void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
[void][Runtime.InteropServices.Marshal]::ReleaseComObject($namespaceShell)
[ordered]@{
    entries = @($entries.ToArray())
    successful_sources = @($successful.ToArray())
    errors = $errors
} | ConvertTo-Json -Depth 6 -Compress
